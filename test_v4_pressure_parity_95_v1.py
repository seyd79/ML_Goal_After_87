from pathlib import Path
from collections import Counter
import csv
import json

from v4_pressure_features_research import (
    B6,
    compute_b6_historical,
)

OUT = Path("outputs/v4_research")
CACHE = Path(
    "outputs/pressure_historical/timeline_cache"
)

GROUPS = [
    (
        "CACHED12",
        OUT / "V4_CACHED12_MANIFEST_V1.csv",
        OUT / "V4_CACHED12_B29_RESEARCH_V1.csv",
        12,
    ),
    (
        "FETCH83",
        OUT / "V4_FETCH83_MANIFEST_V1.csv",
        OUT / "V4_FETCH83_B29_RESEARCH_V1.csv",
        83,
    ),
]


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def field(row, *names):
    for name in names:
        value = row.get(name)

        if value is not None:
            text = str(value).strip()

            if text and text.lower() not in (
                "nan", "none", "null"
            ):
                return text

    return None


def team_info(row):
    home = field(
        row,
        "home_team",
        "home_name",
        "home_team_name",
    )

    away = field(
        row,
        "away_team",
        "away_name",
        "away_team_name",
    )

    if not home or not away:
        raise ValueError(
            "Nom des équipes absent du manifeste. "
            f"Colonnes disponibles : {list(row)}"
        )

    return (
        home,
        away,
        field(row, "home_id", "home_team_id"),
        field(row, "away_id", "away_team_id"),
    )


print("=" * 72)
print("ÉTAPE 214 — TEST DE PARITÉ V4 SUR 95 MATCHS")
print("=" * 72)

total = 0
exact = 0
errors = []
differences = Counter()

for group, manifest_path, ref_path, expected_n in GROUPS:
    manifest = read_rows(manifest_path)
    references = read_rows(ref_path)

    if (
        len(manifest) != expected_n
        or len(references) != expected_n
    ):
        raise SystemExit(
            f"{group} : effectif inattendu"
        )

    ref_by_id = {
        r["match_id"]: r for r in references
    }

    if len(ref_by_id) != expected_n:
        raise SystemExit(
            f"{group} : doublons dans la référence"
        )

    seen = set()
    group_exact = 0

    print(f"\n=== {group} ===")

    for row in manifest:
        mid = row["match_id"]

        if mid in seen:
            raise SystemExit(
                f"{group} : ID dupliqué {mid}"
            )

        seen.add(mid)
        total += 1

        if mid not in ref_by_id:
            errors.append(
                f"{group} {mid} : référence absente"
            )
            continue

        try:
            path = CACHE / f"{mid}.json"

            payload = json.loads(
                path.read_text(encoding="utf-8")
            )

            home, away, home_id, away_id = (
                team_info(row)
            )

            calculated = compute_b6_historical(
                payload,
                home,
                away,
                home_id,
                away_id,
            )

            reference = ref_by_id[mid]
            mismatch = []

            for feature in B6:
                expected = float(reference[feature])
                actual = float(calculated[feature])

                if actual != expected:
                    differences[feature] += 1
                    mismatch.append(
                        f"{feature}: "
                        f"attendu={expected:g}, "
                        f"obtenu={actual:g}"
                    )

            if mismatch:
                errors.append(
                    f"{group} {mid}: "
                    + " | ".join(mismatch)
                )
            else:
                exact += 1
                group_exact += 1

        except Exception as exc:
            errors.append(
                f"{group} {mid}: {exc}"
            )

    print(
        f"Parité complète : "
        f"{group_exact}/{expected_n}"
    )

print("\n=== BILAN GLOBAL ===")
print("Matchs analysés :", total)
print("Matchs identiques sur B6 :", exact)
print("Matchs en erreur :", len(errors))

print("\n=== ÉCARTS PAR VARIABLE ===")

for feature in B6:
    print(
        f"{feature:25s} : "
        f"{differences[feature]}"
    )

if errors:
    print("\n=== PREMIERS ÉCARTS ===")

    for error in errors[:25]:
        print(error)

    print("\nPARITÉ NON VALIDÉE")
else:
    if total != 95:
        raise SystemExit(
            "95 matchs attendus"
        )

    print("\nPARITÉ HISTORIQUE VALIDÉE : 95/95")
    print(
        "Attention : ce résultat ne prouve "
        "pas la compatibilité LIVE."
    )

print("\n=== GARANTIES ===")
print("Aucun appel API.")
print("Aucun label utilisé.")
print("Aucune prédiction.")
print("Aucun fichier de données modifié.")
print("Scanner V3-C inchangé.")
print("ÉTAPE 214 TERMINÉE.")
