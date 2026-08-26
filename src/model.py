"""
model.py — Phase 4: predictive models + uncertainty (leak-free, temporal).

Design
------
Value target = total seasonal fantasy production  target_total = Pv * Fm.
We predict it two ways and keep whichever wins out-of-sample:
  (a) COMPONENT:  E[total] = E[Pv] * E[Fm]   (interpretable; also yields expected minutes/Fm)
  (b) DIRECT:     a single model on target_total

Population: players WITH prior Serie A history (n_hist >= 1). Genuine newcomers (n_hist==0)
are projected separately from priors (newcomers.py) — no ML on players with no history.

Temporal validation (only 3 seasons ⇒ 2 labeled transition folds):
  train  = fold target 2024-25 (features from 2023-24)
  OOS test = fold target 2025-26 (features from 2024-25[+2023-24])
Final fit = both labeled folds combined, then predict 2026-27.

Model comparison per component: Baseline (persistence/role-mean) vs Ridge vs
HistGradientBoosting (vs LightGBM if installed). Best OOS (MAE / Spearman) is chosen.

Uncertainty: HistGradientBoosting quantile regressors (0.1/0.5/0.9) on target_total give
pessimistic / median / optimistic. A composite risk score combines newcomer status, low
minutes, youth, Fm volatility and prediction spread (weights in config).

Outputs: outputs/model_metrics.md, data/processed/predictions.csv
Run: python src/model.py
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

try:
    from . import config as C
    from . import features as F
except ImportError:
    import config as C
    import features as F

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

NUM_FEATURES = [
    "Pv_l1", "Mv_l1", "Fm_l1", "Gf_l1", "Ass_l1", "Amm_l1", "Esp_l1", "Rplus_l1", "Gs_l1",
    "played_l1", "Gf_per_pv_l1", "Ass_per_pv_l1", "Amm_per_pv_l1", "Rplus_per_pv_l1",
    "Pv_l2", "Mv_l2", "Fm_l2", "Gf_l2", "Ass_l2", "played_l2",
    "trend_fm", "trend_pv", "n_hist", "sa_seasons",
    "team_attack", "team_defense", "team_avg_fm", "team_promoted",
]
MACROS = ["P", "D", "C", "A"]


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUM_FEATURES].copy()
    for m in MACROS:
        X[f"macro_{m}"] = (df["macro"] == m).astype(float)
    return X


def _linear_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("m", Ridge(alpha=5.0, random_state=C.RANDOM_STATE))])


def _hgb(quantile: float | None = None):
    kw = dict(max_depth=3, max_iter=300, learning_rate=0.05,
              min_samples_leaf=15, l2_regularization=1.0, random_state=C.RANDOM_STATE)
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(**kw)


def _candidates():
    cands = {"Ridge": _linear_pipe, "HGB": _hgb}
    if HAS_LGBM:
        cands["LGBM"] = lambda: LGBMRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, num_leaves=15,
            min_child_samples=15, reg_lambda=1.0, random_state=C.RANDOM_STATE, verbose=-1)
    return cands


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rho = float(spearmanr(y_true, y_pred).correlation) if len(y_true) > 3 else np.nan
    return {"MAE": mae, "RMSE": rmse, "Spearman": rho, "n": int(len(y_true))}


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def baseline_pv(df):
    role_med = df.groupby("macro")["Pv_l1"].transform("median")
    return df["Pv_l1"].fillna(role_med).fillna(df["Pv_l1"].median()).clip(0, C.MAX_MATCHDAYS)

def baseline_fm(df):
    # weighted persistence, regressed toward role mean
    fm = df["Fm_l1"].fillna(df["Fm_l2"])
    role_mean = df.groupby("macro")["Fm_l1"].transform("mean")
    return (0.7 * fm + 0.3 * role_mean).fillna(role_mean).fillna(6.0)


# ----------------------------------------------------------------------------
# Core training / evaluation
# ----------------------------------------------------------------------------
def _fit_eval(train, test, target_col, played_only):
    """Compare candidates on the OOS fold for one component; return metrics + best name."""
    tr = train[train["n_hist"] >= 1].copy()
    te = test[test["n_hist"] >= 1].copy()
    if played_only:
        tr = tr[tr["target_pv"].fillna(0) > 0]
        te_eval = te[te["target_pv"].fillna(0) > 0]
    else:
        te_eval = te
    Xtr, ytr = _design(tr), tr[target_col].astype(float)
    Xte, yte = _design(te_eval), te_eval[target_col].astype(float)

    results = {}
    # baseline
    if target_col == "target_pv":
        results["Baseline"] = _metrics(yte, baseline_pv(te_eval))
    elif target_col == "target_fm":
        results["Baseline"] = _metrics(yte, baseline_fm(te_eval))
    else:  # target_total
        results["Baseline"] = _metrics(yte, baseline_pv(te_eval) * baseline_fm(te_eval))
    for name, ctor in _candidates().items():
        mdl = ctor()
        mdl.fit(Xtr, ytr)
        results[name] = _metrics(yte, mdl.predict(Xte))
    best = min(results, key=lambda k: results[k]["MAE"])
    return results, best


def _fit_final(labeled, target_col, played_only, model_name):
    tr = labeled[labeled["n_hist"] >= 1].copy()
    if played_only:
        tr = tr[tr["target_pv"].fillna(0) > 0]
    Xtr, ytr = _design(tr), tr[target_col].astype(float)
    if model_name == "Baseline":
        return ("baseline", None)
    mdl = _candidates()[model_name]()
    mdl.fit(Xtr, ytr)
    return (model_name, mdl)


def _predict_component(df, kind, fitted):
    name, mdl = fitted
    if name == "baseline":
        return baseline_pv(df) if kind == "pv" else baseline_fm(df)
    return mdl.predict(_design(df))


# ----------------------------------------------------------------------------
# Newcomer projection (no ML; from priors)
# ----------------------------------------------------------------------------
def project_newcomers(newc: pd.DataFrame, role_fm_median: dict, role_pv_median: dict) -> pd.DataFrame:
    rows = []
    for _, r in newc.iterrows():
        macro = C.ROLE_TO_MACRO.get(r["role"], "C")
        share = r.get("exp_minutes_share")
        is_new = int(r.get("is_newcomer") or 0)
        if pd.notna(share):
            exp_pv = share * C.MAX_MATCHDAYS               # genuine newcomer with a prior
            src = "newcomer_prior"
        elif is_new:
            exp_pv = C.NEWCOMER_MINUTES_DISCOUNT * role_pv_median.get(macro, 20)
            src = "newcomer_prior"
        else:
            # Fringe/returning player with NO recent minutes: was available and passed over
            # → conservative, low expected workload (high uncertainty).
            exp_pv = 0.30 * role_pv_median.get(macro, 20)
            src = "fringe_low_minutes"
        exp_pv = float(np.clip(exp_pv, 0, C.MAX_MATCHDAYS))
        # Fm prior: role median + attacking bonus from goals/assists priors (if any).
        base_fm = role_fm_median.get(macro, 6.0)
        g90, a90 = r.get("goals90_prior"), r.get("assists90_prior")
        bonus = 0.0
        if pd.notna(g90):
            bonus += C.SCORING["goal_A"] * g90 * (exp_pv / max(exp_pv, 1)) * 0.9
        if pd.notna(a90):
            bonus += C.SCORING["assist"] * a90 * 0.9
        exp_fm = base_fm + min(bonus, 2.0)  # cap prior bonus to stay conservative
        exp_total = exp_pv * exp_fm
        conf = str(r.get("prior_confidence", "low"))
        # Wide band for newcomers: scale with (in)confidence.
        spread = {"high": 0.30, "med": 0.45, "low": 0.60}.get(conf, 0.60)
        rows.append({
            "Id": int(r["Id"]), "exp_pv": exp_pv, "exp_fm": exp_fm,
            "exp_total": exp_total,
            "total_p10": exp_total * (1 - spread), "total_p50": exp_total,
            "total_p90": exp_total * (1 + spread),
            "pred_source": src, "prior_confidence": conf,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------
def run() -> str:
    tables = F.build_all()
    train = pd.concat([tables[t] for t in C.OOS_TRAIN_TARGETS], ignore_index=True)
    test = tables[C.OOS_TEST_TARGET]
    predict = tables[C.TARGET_SEASON]
    labeled = pd.concat([tables[t] for t in C.LABELED_TARGETS], ignore_index=True)

    md = ["# Model metrics — Fantacalcio 2026/27\n",
          f"_Temporal OOS fold: train {C.OOS_TRAIN_TARGETS} → test {C.OOS_TEST_TARGET}. "
          f"Stagioni storiche: {C.HISTORICAL_SEASONS}. LightGBM: {HAS_LGBM}._\n",
          "Popolazione di modellazione: giocatori con storia Serie A (n_hist ≥ 1). "
          "I newcomer (0 stagioni) sono proiettati dai priors, non dal modello ML.\n"]

    chosen = {}
    for target_col, played_only, kind in [("target_pv", False, "pv"),
                                          ("target_fm", True, "fm"),
                                          ("target_total", False, "total")]:
        res, best = _fit_eval(train, test, target_col, played_only)
        chosen[target_col] = best
        md.append(f"## {target_col}  (scelto: **{best}**)\n")
        rdf = pd.DataFrame(res).T[["MAE", "RMSE", "Spearman", "n"]].round(3)
        md.append(rdf.to_markdown() + "\n")

    # Decide component vs direct for the point prediction of total, on OOS.
    fit_pv = _fit_final(labeled, "target_pv", False, chosen["target_pv"])
    fit_fm = _fit_final(labeled, "target_fm", True, chosen["target_fm"])
    # OOS comparison: component vs direct
    te = test[test["n_hist"] >= 1]
    comp_pred = np.clip(_predict_component(te, "pv", _fit_final(train, "target_pv", False, chosen["target_pv"])), 0, C.MAX_MATCHDAYS) \
        * _predict_component(te, "fm", _fit_final(train, "target_fm", True, chosen["target_fm"]))
    direct_fit_oos = _fit_final(train, "target_total", False, chosen["target_total"])
    direct_pred = _predict_component(te, "total", direct_fit_oos) if direct_fit_oos[0] != "baseline" \
        else baseline_pv(te) * baseline_fm(te)
    m_comp = _metrics(te["target_total"], comp_pred)
    m_dir = _metrics(te["target_total"], direct_pred)
    use_component = m_comp["MAE"] <= m_dir["MAE"]
    md.append("## Total: COMPONENT (Pv×Fm) vs DIRECT\n")
    md.append(pd.DataFrame({"component": m_comp, "direct": m_dir}).T[["MAE", "RMSE", "Spearman"]].round(3).to_markdown() + "\n")
    md.append(f"**Scelto per il punto: {'COMPONENT (E[Pv]×E[Fm])' if use_component else 'DIRECT'}.**\n")

    # ---- Final predictions for 2026-27 (players with history) ---------------
    hist = predict[predict["n_hist"] >= 1].copy()
    exp_pv = np.clip(_predict_component(hist, "pv", fit_pv), 0, C.MAX_MATCHDAYS)
    exp_fm = _predict_component(hist, "fm", fit_fm)
    if use_component:
        exp_total = exp_pv * exp_fm
    else:
        fit_total = _fit_final(labeled, "target_total", False, chosen["target_total"])
        exp_total = _predict_component(hist, "total", fit_total)
    hist["exp_pv"], hist["exp_fm"], hist["exp_total"] = exp_pv, exp_fm, exp_total

    # Quantile band on total (trained on labeled history population).
    trh = labeled[labeled["n_hist"] >= 1]
    Xtrh, ytrh = _design(trh), trh["target_total"].astype(float)
    Xph = _design(hist)
    for q, col in zip(C.QUANTILES, ["total_p10", "total_p50", "total_p90"]):
        qm = _hgb(quantile=q); qm.fit(Xtrh, ytrh)
        hist[col] = np.clip(qm.predict(Xph), 0, None)
    # enforce monotonicity of quantiles
    hist["total_p50"] = hist[["total_p10", "total_p50"]].max(axis=1)
    hist["total_p90"] = hist[["total_p50", "total_p90"]].max(axis=1)
    hist["pred_source"] = "model"
    hist["prior_confidence"] = ""

    # ---- Newcomer projections ----------------------------------------------
    role_fm_median = trh.groupby("macro")["target_fm"].median().to_dict()
    role_pv_median = trh[trh["target_pv"] > 0].groupby("macro")["target_pv"].median().to_dict()
    newc = predict[predict["n_hist"] < 1].drop(
        columns=["prior_confidence"], errors="ignore").copy()
    proj = project_newcomers(predict[predict["n_hist"] < 1], role_fm_median, role_pv_median)
    newc = newc.merge(proj, on="Id", how="left")

    # ---- Assemble predictions ----------------------------------------------
    keep = ["Id", "Nome", "team", "role", "macro", "exp_pv", "exp_fm", "exp_total",
            "total_p10", "total_p50", "total_p90", "pred_source", "prior_confidence",
            "is_newcomer"]
    allpred = pd.concat([hist[keep], newc[keep]], ignore_index=True)

    # ---- Component detail (goals/assists expected) for reporting ------------
    # Expected goals/assists ≈ per-pv rate (lagged, regressed) * exp_pv.
    def _exp_rate(df, col):
        r = df.get(col)
        return r.fillna(df.groupby("macro")[col].transform("median")) if r is not None else np.nan
    ph = predict.set_index("Id")
    allpred = allpred.merge(
        ph[["Gf_per_pv_l1", "Ass_per_pv_l1"]].reset_index(), on="Id", how="left")
    allpred["exp_goals"] = (allpred["Gf_per_pv_l1"].fillna(0) * allpred["exp_pv"]).round(1)
    allpred["exp_assists"] = (allpred["Ass_per_pv_l1"].fillna(0) * allpred["exp_pv"]).round(1)

    # ---- Risk score ---------------------------------------------------------
    allpred = _risk_score(allpred, predict)

    allpred = allpred.sort_values("exp_total", ascending=False).reset_index(drop=True)
    out = os.path.join(C.PROCESSED_DIR, "predictions.csv")
    allpred.round(3).to_csv(out, index=False, encoding="utf-8")

    # ---- Feature importance (permutation, OOS, on chosen total path) --------
    md.append("## Feature importance (permutation, OOS fold)\n")
    imp_model = _fit_final(train, "target_total", False,
                           chosen["target_total"] if chosen["target_total"] != "Baseline" else "HGB")
    if imp_model[0] != "baseline":
        te2 = test[test["n_hist"] >= 1]
        pi = permutation_importance(imp_model[1], _design(te2), te2["target_total"].astype(float),
                                    n_repeats=8, random_state=C.RANDOM_STATE)
        imp = (pd.DataFrame({"feature": _design(te2).columns, "importance": pi.importances_mean})
               .sort_values("importance", ascending=False).head(15).round(3))
        md.append(imp.to_markdown(index=False) + "\n")

    md.append("## Note metodologiche\n")
    md.append("- Solo 2 coppie di transizione: metriche OOS indicative; preferiti modelli semplici se competitivi.\n")
    md.append("- `exp_goals`/`exp_assists` sono stime derivate dai tassi laggati (no xG/xA nei dati storici).\n")
    md.append("- I newcomer usano priors (web/fallback); confidenza in `prior_confidence`.\n")

    with open(os.path.join(C.OUTPUTS_DIR, "model_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[model] scelte: {chosen} | usa_componente={use_component}")
    print(f"[model] predictions -> {out} ({len(allpred)} giocatori, "
          f"{int(allpred['is_newcomer'].sum())} newcomer)")
    print(f"[model] metrics -> {os.path.join(C.OUTPUTS_DIR, 'model_metrics.md')}")
    return out


def _risk_score(allpred: pd.DataFrame, predict: pd.DataFrame) -> pd.DataFrame:
    p = predict.set_index("Id")
    df = allpred.copy()
    w = C.RISK_WEIGHTS
    # newcomer
    r_new = df["is_newcomer"].fillna(0).astype(float)
    # low minutes: low expected Pv
    r_min = (1 - (df["exp_pv"] / C.MAX_MATCHDAYS)).clip(0, 1)
    # young age (from priors if present)
    age = df["Id"].map(p["age"]) if "age" in p.columns else pd.Series(np.nan, index=df.index)
    r_young = (age < C.YOUNG_AGE_THRESHOLD).astype(float).fillna(0)
    # Fm volatility across lag seasons
    fm1 = df["Id"].map(p["Fm_l1"]); fm2 = df["Id"].map(p["Fm_l2"])
    r_vol = (fm1 - fm2).abs().fillna(fm1.std() if fm1.std() == fm1.std() else 0)
    r_vol = (r_vol / (r_vol.max() or 1)).clip(0, 1)
    # prediction spread
    spread = ((df["total_p90"] - df["total_p10"]) / (df["total_p50"].replace(0, np.nan))).abs()
    r_spread = (spread / (spread.max() or 1)).fillna(0.5).clip(0, 1)
    risk = (w["newcomer"] * r_new + w["low_minutes"] * r_min + w["young_age"] * r_young
            + w["fm_volatility"] * r_vol + w["prediction_spread"] * r_spread)
    df["risk"] = (risk / sum(w.values())).round(3)
    return df


if __name__ == "__main__":
    run()
