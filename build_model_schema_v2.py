#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_model_schema_v2.py

CONSTRUCTION DU SCHÉMA OFFICIEL DU NOUVEAU MODÈLE
=================================================

OBJECTIF
--------
À partir du dataset reconstruit propre et du rapport de couverture,
ce script :

1) classe les features en CORE / CONDITIONAL / DROP
2) applique des exclusions sémantiques connues
3) évite les doublons évidents du vieux schéma
4) construit un nouveau dataset d'entraînement basé uniquement sur le schéma V2
5) produit un JSON de schéma officiel réutilisable par le futur train_model_v2.py
6) ne modifie PAS le scanner de production

ENTRÉES PAR DÉFAUT
------------------
Dataset reconstruit :
    outputs/rebuilt_dataset_v2/Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv

Couverture :
    outputs/rebuilt_dataset_v2/feature_coverage_v2.csv

SORTIES
-------
outputs/model_schema_v2/
├── model_schema_v2.json
├── model_schema_v2_features.csv
├── Training_Dataset_Model_V2.csv
├── Training_Dataset_Model_V2_COMPLETE.csv
├── model_schema_v2_summary.csv
└── model_schema_v2_report.md

RÈGLES
------
CORE:
    coverage >= 95%

CONDITIONAL:
    80% <= coverage < 95%

DROP:
    coverage < 80%

EXCLUSIONS SÉMANTIQUES PAR DÉFAUT
---------------------------------
- doublons de standings:
    home_position
    away_position
    home_points
    away_points

- marchés secondaires de cotes souvent indisponibles:
    odds_over05
    odds_over15
    odds_over25
    odds_over35
    odds_under25
    odds_btts_yes
    odds_btts_no

- H2H sous couverture insuffisante:
    h2h5_over25_freq
    h2h5_btts_freq

- league_2nd_half_over15_freq:
    exclue du schéma CORE par défaut car coûteuse à reconstruire.
    Peut être réactivée avec --include-expensive

- xG:
    conservé seulement s'il atteint le seuil CORE,
    sinon classé CONDITIONAL. Pas d'imputation cachée.

IMPORTANT
---------
Ce script ne force AUCUN 96/96.
Il construit le schéma autour des données réellement fiables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = (
    BASE_DIR
    / "outputs"
    / "rebuilt_dataset_v2"
    / "Clean_Dataset_Goal_After_87_REBUILT_RAW_v2.csv"
)

DEFAULT_COVERAGE = (
    BASE_DIR
    / "outputs"
    / "rebuilt_dataset_v2"
    / "feature_coverage_v2.csv"
)

DEFAULT_OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "model_schema_v2"
)

TARGET_COLUMN = "goal_after_87"
ID_COLUMN = "match_id"

KNOWN_DUPLICATES = {
    "home_position",
    "away_position",
    "home_points",
    "away_points",
}

KNOWN_SECONDARY_ODDS = {
    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",
}

KNOWN_H2H_LOW_COVERAGE = {
    "h2h5_over25_freq",
    "h2h5_btts_freq",
}

EXPENSIVE_FEATURES = {
    "league_2nd_half_over15_freq",
}

PREFERRED_METADATA = [
    "match_id",
    "competition_id",
    "season_id",
    "home_team",
    "away_team",
    "kickoff_utc",
    "captured_at_utc",
    "captured_at",
    "target_status",
    "target_source",
    "target_resolution_class",
    "resolution_class",
    "target_confidence",
    "goal_minutes_after_87",
    "valid_goal_minutes_after_87",
]


def resolve_path(
    explicit: str | None,
    default_path: Path,
) -> Path:
    path = (
        Path(explicit).expanduser()
        if explicit
        else default_path
    )

    if not path.is_absolute():
        path = (
            BASE_DIR
            / path
        )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    return path


def classify_by_coverage(
    coverage: float,
    core_threshold: float,
    conditional_threshold: float,
) -> str:
    if coverage >= core_threshold:
        return "CORE"

    if coverage >= conditional_threshold:
        return "CONDITIONAL"

    return "DROP"


def semantic_override(
    feature: str,
    base_class: str,
    include_expensive: bool,
) -> tuple[str, str]:
    """
    Retourne (classification finale, raison).
    """

    if feature in KNOWN_DUPLICATES:
        return (
            "DROP",
            "duplicate_semantic_alias",
        )

    if feature in KNOWN_SECONDARY_ODDS:
        return (
            "DROP",
            "secondary_odds_unreliable_or_sparse",
        )

    if feature in KNOWN_H2H_LOW_COVERAGE:
        return (
            "DROP",
            "h2h_sparse",
        )

    if (
        feature in EXPENSIVE_FEATURES
        and not include_expensive
    ):
        return (
            "DROP",
            "expensive_historical_reconstruction",
        )

    return (
        base_class,
        "coverage_rule",
    )


def build_feature_table(
    coverage_df: pd.DataFrame,
    core_threshold: float,
    conditional_threshold: float,
    include_expensive: bool,
) -> pd.DataFrame:
    required = {
        "feature",
        "group",
        "available",
        "missing",
        "coverage_rate",
    }

    missing = required - set(
        coverage_df.columns
    )

    if missing:
        raise ValueError(
            "Colonnes manquantes dans feature_coverage_v2.csv : "
            + ", ".join(
                sorted(missing)
            )
        )

    rows: List[Dict] = []

    for _, row in coverage_df.iterrows():
        feature = str(
            row["feature"]
        )

        group = str(
            row["group"]
        )

        coverage = float(
            row["coverage_rate"]
        )

        base_class = classify_by_coverage(
            coverage=coverage,
            core_threshold=core_threshold,
            conditional_threshold=conditional_threshold,
        )

        final_class, reason = semantic_override(
            feature=feature,
            base_class=base_class,
            include_expensive=include_expensive,
        )

        rows.append({
            "feature":
                feature,
            "group":
                group,
            "available":
                int(
                    row["available"]
                ),
            "missing":
                int(
                    row["missing"]
                ),
            "coverage_rate":
                coverage,
            "base_class":
                base_class,
            "final_class":
                final_class,
            "decision_reason":
                reason,
        })

    out = pd.DataFrame(
        rows
    )

    order = {
        "CORE": 0,
        "CONDITIONAL": 1,
        "DROP": 2,
    }

    out["_order"] = out[
        "final_class"
    ].map(
        order
    )

    out = (
        out
        .sort_values(
            by=[
                "_order",
                "group",
                "coverage_rate",
                "feature",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return out


def valid_binary_target(
    df: pd.DataFrame,
) -> pd.Series:
    target = pd.to_numeric(
        df[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    return target.isin(
        [0, 1]
    )


def metadata_columns(
    df: pd.DataFrame,
) -> List[str]:
    return [
        column
        for column in PREFERRED_METADATA
        if column in df.columns
    ]


def build_training_dataset(
    source_df: pd.DataFrame,
    core_features: Sequence[str],
) -> pd.DataFrame:
    df = source_df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{TARGET_COLUMN} absent du dataset."
        )

    missing_core = [
        feature
        for feature in core_features
        if feature not in df.columns
    ]

    if missing_core:
        raise ValueError(
            "Features CORE absentes du dataset : "
            + ", ".join(
                missing_core
            )
        )

    df = df[
        valid_binary_target(
            df
        )
    ].copy()

    df[
        TARGET_COLUMN
    ] = pd.to_numeric(
        df[
            TARGET_COLUMN
        ],
        errors="coerce",
    ).astype(
        int
    )

    columns = (
        metadata_columns(
            df
        )
        + list(
            core_features
        )
        + [
            TARGET_COLUMN
        ]
    )

    columns = list(
        dict.fromkeys(
            columns
        )
    )

    return df[
        columns
    ].copy()


def make_report(
    feature_table: pd.DataFrame,
    training_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    core_threshold: float,
    conditional_threshold: float,
    include_expensive: bool,
) -> str:
    core = feature_table[
        feature_table[
            "final_class"
        ]
        == "CORE"
    ]

    conditional = feature_table[
        feature_table[
            "final_class"
        ]
        == "CONDITIONAL"
    ]

    dropped = feature_table[
        feature_table[
            "final_class"
        ]
        == "DROP"
    ]

    core_pre = int(
        (
            core[
                "group"
            ]
            == "PRE"
        ).sum()
    )

    core_live = int(
        (
            core[
                "group"
            ]
            == "LIVE"
        ).sum()
    )

    positives = int(
        (
            training_df[
                TARGET_COLUMN
            ]
            == 1
        ).sum()
    )

    negatives = int(
        (
            training_df[
                TARGET_COLUMN
            ]
            == 0
        ).sum()
    )

    lines = [
        "# Model Schema V2",
        "",
        "## Seuils",
        "",
        f"- CORE : couverture >= {core_threshold:.0%}",
        f"- CONDITIONAL : {conditional_threshold:.0%} <= couverture < {core_threshold:.0%}",
        f"- DROP : couverture < {conditional_threshold:.0%}",
        "",
        "## Résumé",
        "",
        f"- Features CORE : **{len(core)}**",
        f"- PRE CORE : **{core_pre}**",
        f"- LIVE CORE : **{core_live}**",
        f"- Features CONDITIONAL : **{len(conditional)}**",
        f"- Features DROP : **{len(dropped)}**",
        f"- Lignes avec target résolu : **{len(training_df)}**",
        f"- Lignes complètes sur le schéma CORE : **{len(complete_df)}**",
        f"- Positifs : **{positives}**",
        f"- Négatifs : **{negatives}**",
        "",
        "## Politique",
        "",
        "- Aucun remplissage médian automatique n'est effectué ici.",
        "- Les features secondaires de cotes sont retirées du schéma initial.",
        "- Les alias standings dupliqués sont retirés.",
        "- Les H2H faibles en couverture sont retirés.",
        (
            "- `league_2nd_half_over15_freq` est incluse."
            if include_expensive
            else "- `league_2nd_half_over15_freq` est retirée par défaut car coûteuse à reconstruire."
        ),
        "- Le dataset COMPLETE contient uniquement les lignes sans NaN sur toutes les features CORE.",
        "",
        "## Features CORE",
        "",
    ]

    for _, row in core.iterrows():
        lines.append(
            f"- `{row['feature']}` "
            f"({row['group']}, "
            f"{row['coverage_rate']:.2%})"
        )

    lines.extend([
        "",
        "## Features CONDITIONAL",
        "",
    ])

    if conditional.empty:
        lines.append(
            "- Aucune."
        )
    else:
        for _, row in conditional.iterrows():
            lines.append(
                f"- `{row['feature']}` "
                f"({row['group']}, "
                f"{row['coverage_rate']:.2%})"
            )

    lines.extend([
        "",
        "## Features DROP",
        "",
    ])

    if dropped.empty:
        lines.append(
            "- Aucune."
        )
    else:
        for _, row in dropped.iterrows():
            lines.append(
                f"- `{row['feature']}` "
                f"({row['group']}, "
                f"{row['coverage_rate']:.2%}) "
                f"— {row['decision_reason']}"
            )

    lines.extend([
        "",
        "## Prochaine étape",
        "",
        "Auditer `Training_Dataset_Model_V2_COMPLETE.csv` avant entraînement.",
        "Ensuite seulement construire `train_model_v2.py`.",
    ])

    return "\n".join(
        lines
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Construit le schéma officiel du nouveau modèle V2."
        )
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset reconstruit RAW."
        ),
    )

    parser.add_argument(
        "--coverage",
        default=None,
        help=(
            "feature_coverage_v2.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Dossier de sortie."
        ),
    )

    parser.add_argument(
        "--core-threshold",
        type=float,
        default=0.95,
        help=(
            "Seuil CORE. Défaut 0.95."
        ),
    )

    parser.add_argument(
        "--conditional-threshold",
        type=float,
        default=0.80,
        help=(
            "Seuil CONDITIONAL. Défaut 0.80."
        ),
    )

    parser.add_argument(
        "--include-expensive",
        action="store_true",
        help=(
            "Autorise les features coûteuses comme league_2nd_half_over15_freq."
        ),
    )

    args = parser.parse_args()

    dataset_path = resolve_path(
        args.dataset,
        DEFAULT_DATASET,
    )

    coverage_path = resolve_path(
        args.coverage,
        DEFAULT_COVERAGE,
    )

    output_dir = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print(
        "BUILD MODEL SCHEMA V2"
    )
    print("=" * 100)

    print(
        f"Dataset : {dataset_path}"
    )

    print(
        f"Coverage : {coverage_path}"
    )

    source_df = pd.read_csv(
        dataset_path
    )

    coverage_df = pd.read_csv(
        coverage_path
    )

    feature_table = build_feature_table(
        coverage_df=coverage_df,
        core_threshold=args.core_threshold,
        conditional_threshold=args.conditional_threshold,
        include_expensive=args.include_expensive,
    )

    core_features = (
        feature_table[
            feature_table[
                "final_class"
            ]
            == "CORE"
        ][
            "feature"
        ]
        .tolist()
    )

    conditional_features = (
        feature_table[
            feature_table[
                "final_class"
            ]
            == "CONDITIONAL"
        ][
            "feature"
        ]
        .tolist()
    )

    dropped_features = (
        feature_table[
            feature_table[
                "final_class"
            ]
            == "DROP"
        ][
            "feature"
        ]
        .tolist()
    )

    training_df = build_training_dataset(
        source_df=source_df,
        core_features=core_features,
    )

    if core_features:
        complete_mask = (
            training_df[
                core_features
            ]
            .notna()
            .all(
                axis=1
            )
        )

        complete_df = training_df[
            complete_mask
        ].copy()
    else:
        complete_df = training_df.iloc[
            0:0
        ].copy()

    schema = {
        "schema_version":
            "model_schema_v2",
        "generated_from":
            {
                "dataset":
                    str(
                        dataset_path
                    ),
                "coverage":
                    str(
                        coverage_path
                    ),
            },
        "thresholds":
            {
                "core":
                    args.core_threshold,
                "conditional":
                    args.conditional_threshold,
            },
        "include_expensive":
            bool(
                args.include_expensive
            ),
        "target_column":
            TARGET_COLUMN,
        "id_column":
            ID_COLUMN,
        "core_features":
            core_features,
        "conditional_features":
            conditional_features,
        "dropped_features":
            dropped_features,
        "feature_count":
            {
                "core":
                    len(
                        core_features
                    ),
                "conditional":
                    len(
                        conditional_features
                    ),
                "drop":
                    len(
                        dropped_features
                    ),
            },
    }

    schema_path = (
        output_dir
        / "model_schema_v2.json"
    )

    features_path = (
        output_dir
        / "model_schema_v2_features.csv"
    )

    training_path = (
        output_dir
        / "Training_Dataset_Model_V2.csv"
    )

    complete_path = (
        output_dir
        / "Training_Dataset_Model_V2_COMPLETE.csv"
    )

    summary_path = (
        output_dir
        / "model_schema_v2_summary.csv"
    )

    report_path = (
        output_dir
        / "model_schema_v2_report.md"
    )

    schema_path.write_text(
        json.dumps(
            schema,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    feature_table.to_csv(
        features_path,
        index=False,
        encoding="utf-8-sig",
    )

    training_df.to_csv(
        training_path,
        index=False,
        encoding="utf-8-sig",
    )

    complete_df.to_csv(
        complete_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([
        {
            "metric":
                "source_rows",
            "value":
                len(
                    source_df
                ),
        },
        {
            "metric":
                "resolved_target_rows",
            "value":
                len(
                    training_df
                ),
        },
        {
            "metric":
                "complete_core_rows",
            "value":
                len(
                    complete_df
                ),
        },
        {
            "metric":
                "core_features",
            "value":
                len(
                    core_features
                ),
        },
        {
            "metric":
                "conditional_features",
            "value":
                len(
                    conditional_features
                ),
        },
        {
            "metric":
                "dropped_features",
            "value":
                len(
                    dropped_features
                ),
        },
        {
            "metric":
                "positive_rows",
            "value":
                int(
                    (
                        training_df[
                            TARGET_COLUMN
                        ]
                        == 1
                    ).sum()
                ),
        },
        {
            "metric":
                "negative_rows",
            "value":
                int(
                    (
                        training_df[
                            TARGET_COLUMN
                        ]
                        == 0
                    ).sum()
                ),
        },
    ])

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = make_report(
        feature_table=feature_table,
        training_df=training_df,
        complete_df=complete_df,
        core_threshold=args.core_threshold,
        conditional_threshold=args.conditional_threshold,
        include_expensive=args.include_expensive,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    core_pre = int(
        (
            feature_table[
                feature_table[
                    "final_class"
                ]
                == "CORE"
            ][
                "group"
            ]
            == "PRE"
        ).sum()
    )

    core_live = int(
        (
            feature_table[
                feature_table[
                    "final_class"
                ]
                == "CORE"
            ][
                "group"
            ]
            == "LIVE"
        ).sum()
    )

    print()
    print("-" * 100)

    print(
        f"Features CORE : "
        f"{len(core_features)}"
    )

    print(
        f"  PRE CORE : "
        f"{core_pre}"
    )

    print(
        f"  LIVE CORE : "
        f"{core_live}"
    )

    print(
        f"Features CONDITIONAL : "
        f"{len(conditional_features)}"
    )

    print(
        f"Features DROP : "
        f"{len(dropped_features)}"
    )

    print(
        f"Lignes target résolu : "
        f"{len(training_df)}"
    )

    print(
        f"Lignes complètes CORE : "
        f"{len(complete_df)}"
    )

    if len(training_df):

        print(
            f"Target rate : "
            f"{training_df[TARGET_COLUMN].mean():.2%}"
        )

    print("-" * 100)

    print(
        f"Schema JSON : "
        f"{schema_path}"
    )

    print(
        f"Features CSV : "
        f"{features_path}"
    )

    print(
        f"Dataset V2 : "
        f"{training_path}"
    )

    print(
        f"Dataset V2 COMPLETE : "
        f"{complete_path}"
    )

    print(
        f"Rapport : "
        f"{report_path}"
    )

    print("=" * 100)

    print()
    print(
        "Le scanner de production n'a pas été modifié."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
