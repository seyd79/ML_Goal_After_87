#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

ROOT = Path("outputs/v4_research")

TRAIN = ROOT / "V4_TRAIN278_B29_RESEARCH_V1.csv"
OOF = ROOT / "V4_SIMPLE_CHRONO_OOF_V1.csv"

print("=" * 72)
print("ÉTAPE 225 — DIVERSITÉ TEMPORELLE V4.1")
print("=" * 72)

train = pd.read_csv(TRAIN, dtype={"match_id": str})
oof = pd.read_csv(OOF, dtype={"match_id": str})

if len(train) != 278 or len(oof) != 121:
    raise SystemExit("Effectifs inattendus")

if (
    train.match_id.duplicated().any()
    or oof.match_id.duplicated().any()
):
    raise SystemExit("Identifiants dupliqués")

if not set(oof.match_id).issubset(set(train.match_id)):
    raise SystemExit("OOF non inclus dans TRAIN")

train["date"] = pd.to_datetime(
    train.kickoff_utc,
    utc=True,
    errors="raise"
).dt.strftime("%Y-%m-%d")

oof["date"] = pd.to_datetime(
    oof.kickoff_utc,
    utc=True,
    errors="raise"
).dt.strftime("%Y-%m-%d")

train["goal_after_87"] = pd.to_numeric(
    train.goal_after_87,
    errors="raise"
)

if not train.goal_after_87.isin([0, 1]).all():
    raise SystemExit("Labels invalides")

print("\n=== TRAIN278 PAR JOURNÉE ===")

daily = train.groupby("date").agg(
    matchs=("match_id", "size"),
    buts_apres_87=("goal_after_87", "sum")
)

daily["taux"] = (
    daily.buts_apres_87 / daily.matchs * 100
).round(1)

print(daily.to_string())

print("\n=== OOF121 PAR JOURNÉE ===")
print(
    oof.groupby("date")
    .size()
    .rename("matchs")
    .to_string()
)

print("\n=== DIVERSITÉ ===")

days = train.date.nunique()
top2 = train.date.value_counts().head(2).sum()

print("Matchs TRAIN :", len(train))
print("Journées TRAIN :", days)
print("Journées OOF :", oof.date.nunique())

print(
    "Concentration sur les 2 journées "
    f"les plus représentées : {top2}/278 "
    f"= {top2 / 278:.1%}"
)

print("\n=== SOURCES HISTORIQUES ===")

print(
    train.groupby("research_origin")
    .agg(
        matchs=("match_id", "size"),
        journees=("date", "nunique")
    )
    .to_string()
)

print("\n=== CONCLUSION ===")
print("Audit descriptif uniquement.")
print("Aucun modèle entraîné.")
print("Aucun fichier modifié.")
print("Aucun appel API.")
print("VALIDATION non consultée.")
print("TEST_FUTUR non consulté.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 225 TERMINÉE.")
