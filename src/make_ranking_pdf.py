"""Genera outputs/classifica_attesa.pdf con la classifica squadre per fantapunti attesi."""
from __future__ import annotations
import os
import pandas as pd

import analyze_rosters as A
import config as C

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

GREEN = colors.HexColor("#0b3d2e")
LIGHT = colors.HexColor("#e7f5ec")
GREY = colors.HexColor("#f2f2f2")


def build_rows():
    pv = pd.read_csv(os.path.join(C.PROCESSED_DIR, "player_values.csv"))
    teams = A.parse_rosters(A.ROSTERS_FILE)
    df, _ = A.match_players(teams, pv)
    rows = []
    for team in teams:
        tp = df[df["team"] == team].to_dict("records")
        bm = A.best_module(tp)
        rows.append({"team": team, "mod": bm["module"], "xi": round(bm["val"]),
                     "depth": round(sum(p["exp_total"] for p in tp)),
                     "n": len(tp), "spesa": sum(p["cost"] for p in tp), "bm": bm, "players": tp})
    rows.sort(key=lambda r: -r["xi"])
    # depth rank
    dorder = sorted(rows, key=lambda r: -r["depth"])
    for i, r in enumerate(dorder):
        r["depth_rank"] = i + 1
    return rows


def run():
    rows = build_rows()
    out = os.path.join(C.OUTPUTS_DIR, "classifica_attesa.pdf")
    doc = SimpleDocTemplate(out, pagesize=A4, topMargin=16*mm, bottomMargin=14*mm,
                            leftMargin=14*mm, rightMargin=14*mm,
                            title="Classifica Attesa - Fantacalcio Mantra 2026/27")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=GREEN, fontSize=18, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#555"))
    note = ParagraphStyle("note", parent=ss["Normal"], fontSize=8.5, textColor=colors.HexColor("#444"), spaceBefore=8)
    story = []
    story.append(Paragraph("Classifica Attesa &mdash; Fantacalcio Mantra 2026/27", h1))
    story.append(Paragraph("Lega ISACA (FANTAISACA) &middot; 12 squadre &middot; proiezioni ML + probabili formazioni + infortuni", sub))
    story.append(Spacer(1, 8))

    header = ["#", "Squadra", "Modulo", "Fantapunti XI", "Profondità rosa", "Giocatori", "Spesa"]
    data = [header]
    my_row_idx = None
    for i, r in enumerate(rows):
        mine = (r["team"] == C.MY_TEAM)
        if mine:
            my_row_idx = i + 1
        data.append([str(i + 1),
                     r["team"] + ("  (tu)" if mine else ""),
                     r["mod"], f"{r['xi']}", f"{r['depth']}  ({r['depth_rank']}°)",
                     str(r["n"]), str(r["spesa"])])

    tbl = Table(data, colWidths=[9*mm, 52*mm, 20*mm, 26*mm, 34*mm, 20*mm, 18*mm], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GREY]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, GREEN),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]
    if my_row_idx:
        style.append(("BACKGROUND", (0, my_row_idx), (-1, my_row_idx), LIGHT))
        style.append(("FONTNAME", (0, my_row_idx), (-1, my_row_idx), "Helvetica-Bold"))
    tbl.setStyle(TableStyle(style))
    story.append(tbl)

    # my team formation
    me = next((r for r in rows if r["team"] == C.MY_TEAM), None)
    if me:
        story.append(Paragraph(
            f"<b>{C.MY_TEAM}</b> &mdash; {my_row_idx}° per XI, ma <b>1° per profondità rosa</b> "
            f"({me['depth']} pt): un vantaggio pesante su 38 giornate (infortuni, rotazioni, asta di febbraio). "
            f"Modulo migliore: <b>{me['mod']}</b>.", note))
        xi = []
        for s in me["bm"]["slots"]:
            if s["p"]:
                xi.append(f"{'/'.join(sorted(s['roles']))}: <b>{s['p']['Nome']}</b> ({s['p']['exp_total']:.0f})")
        story.append(Paragraph("<b>Formazione tipo:</b> " + " &nbsp;&bull;&nbsp; ".join(xi), note))

    story.append(Paragraph(
        "<i>Metodo:</i> \"Fantapunti XI\" = somma dei fantapunti stagionali attesi degli 11 titolari nel modulo ottimale "
        "(presenze attese × fantamedia + bonus rigori/piazzati, al netto degli infortuni noti). "
        "\"Profondità rosa\" = somma su tutta la rosa. Stime, non certezze.", note))

    doc.build(story)
    print(f"[pdf] creato {out}")
    return out


if __name__ == "__main__":
    run()
