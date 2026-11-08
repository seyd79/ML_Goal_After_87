"""
verify_dataset.py

Vérification complète du dataset ML avant validation temporelle.

Dataset attendu :
    ML_Dataset_PreMatch_Live87.csv

Objectif :
    Vérifier que le dataset est propre, cohérent et exploitable
    pour entraîner un modèle pré-match + état live à la 87e minute
    afin de prédire goal_after_87.

IMPORTANT :
    Ce script ne modifie PAS le dataset original.
    Il produit uniquement un rapport de vérification.

Sorties :
    outputs/dataset_verification/
        verification_report.txt
        verification_summary.json
        missing_values.csv
        duplicate_match_ids.csv
        suspicious_columns.csv
        column_profile.csv
"""

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("ML_Dataset_PreMatch_Live87.csv")
OUTPUT_DIR = Path("outputs/dataset_verification")

TARGET = "goal_after_87"
MATCH_ID = "match_id"

EXPECTED_ROWS = 13800
EXPECTED_COLUMNS = 98


# ============================================================
# UTILITAIRES
# ============================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_float(value):
    if pd.isna(value):
        return None

    try:
        return float(value)
    except Exception:
        return None


def detect_date_columns(df):
    """
    Détecte les colonnes qui semblent être des dates/datetimes.
    """

    candidates = []

    date_name_patterns = [
        "date",
        "match_date",
        "game_date",
        "fixture_date",
        "kickoff",
        "datetime",
        "timestamp",
        "time",
        "utc",
        "start_time",
        "match_time",
    ]

    for column in df.columns:
        column_lower = column.lower()

        # Détection par nom
        name_match = any(
            pattern in column_lower
            for pattern in date_name_patterns
        )

        if name_match:
            candidates.append(column)
            continue

        # Détection par dtype
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            candidates.append(column)
            continue

        # Tentative de parsing seulement pour les colonnes object
        if df[column].dtype == "object":
            sample = df[column].dropna().astype(str).head(500)

            if len(sample) == 0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                parsed = pd.to_datetime(
                    sample,
                    errors="coerce"
                )

            success_rate = parsed.notna().mean()

            if success_rate >= 0.80:
                candidates.append(column)

    return list(dict.fromkeys(candidates))


def analyze_date_column(df, column):
    """
    Analyse une colonne de date et vérifie son ordre chronologique.
    """

    result = {
        "column": column,
        "parse_success_rate": None,
        "min_date": None,
        "max_date": None,
        "is_monotonic_increasing": None,
        "is_monotonic_decreasing": None,
        "unique_dates": None,
        "duplicate_dates": None,
        "chronological_order": "UNKNOWN",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        parsed = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    valid = parsed.notna()

    if valid.sum() == 0:
        return result

    result["parse_success_rate"] = round(
        float(valid.mean()),
        6
    )

    valid_dates = parsed[valid]

    result["min_date"] = str(valid_dates.min())
    result["max_date"] = str(valid_dates.max())

    result["unique_dates"] = int(valid_dates.nunique())

    result["duplicate_dates"] = int(
        valid_dates.duplicated().sum()
    )

    # Vérification sur les lignes valides uniquement
    ordered = valid_dates.reset_index(drop=True)

    result["is_monotonic_increasing"] = bool(
        ordered.is_monotonic_increasing
    )

    result["is_monotonic_decreasing"] = bool(
        ordered.is_monotonic_decreasing
    )

    if result["is_monotonic_increasing"]:
        result["chronological_order"] = "OLDEST_TO_NEWEST"

    elif result["is_monotonic_decreasing"]:
        result["chronological_order"] = "NEWEST_TO_OLDEST"

    else:
        result["chronological_order"] = "NOT_SORTED"

    return result


def detect_suspicious_columns(df):
    """
    Recherche des colonnes potentiellement problématiques
    pour le ML.

    On ne supprime rien automatiquement.
    """

    suspicious = []

    leakage_keywords = [
        "final",
        "result",
        "winner",
        "loser",
        "fulltime",
        "ft_",
        "_ft",
        "after",
        "post",
        "final_score",
        "goals_after",
        "goal_after",
        "minute_goal",
        "last_goal",
        "match_end",
    ]

    identifier_keywords = [
        "id",
        "uuid",
        "fixture",
        "match_id",
    ]

    for column in df.columns:
        column_lower = column.lower()

        reasons = []
        severity = "INFO"

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        if any(
            keyword in column_lower
            for keyword in identifier_keywords
        ):
            reasons.append(
                "possible_identifier"
            )

        # ----------------------------------------------------
        # Leakage
        # ----------------------------------------------------

        leakage_matches = [
            keyword
            for keyword in leakage_keywords
            if keyword in column_lower
        ]

        if leakage_matches:
            reasons.append(
                "possible_target_leakage:" +
                ",".join(leakage_matches)
            )
            severity = "HIGH"

        # ----------------------------------------------------
        # Constante
        # ----------------------------------------------------

        nunique = df[column].nunique(dropna=False)

        if nunique <= 1:
            reasons.append("constant_column")

            if severity != "HIGH":
                severity = "MEDIUM"

        # ----------------------------------------------------
        # Quasi-identifiant
        # ----------------------------------------------------

        unique_ratio = (
            df[column].nunique(dropna=True)
            / max(len(df), 1)
        )

        if unique_ratio >= 0.99:
            reasons.append(
                "almost_unique_column"
            )

            if severity == "INFO":
                severity = "MEDIUM"

        if reasons:
            suspicious.append({
                "column": column,
                "severity": severity,
                "unique_values": int(nunique),
                "unique_ratio": round(
                    float(unique_ratio),
                    6
                ),
                "reasons": "; ".join(reasons),
            })

    return pd.DataFrame(suspicious)


def build_column_profile(df):
    """
    Profil détaillé de chaque colonne.
    """

    rows = []

    for column in df.columns:

        series = df[column]

        missing_count = int(series.isna().sum())

        missing_pct = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0
        )

        unique_count = int(
            series.nunique(dropna=True)
        )

        unique_ratio = (
            unique_count / len(df)
            if len(df) > 0
            else 0
        )

        rows.append({
            "column": column,
            "dtype": str(series.dtype),
            "missing_count": missing_count,
            "missing_pct": round(
                missing_pct,
                4
            ),
            "unique_count": unique_count,
            "unique_ratio": round(
                unique_ratio,
                6
            ),
            "min": (
                safe_float(series.min())
                if pd.api.types.is_numeric_dtype(series)
                else None
            ),
            "max": (
                safe_float(series.max())
                if pd.api.types.is_numeric_dtype(series)
                else None
            ),
            "mean": (
                safe_float(series.mean())
                if pd.api.types.is_numeric_dtype(series)
                else None
            ),
            "median": (
                safe_float(series.median())
                if pd.api.types.is_numeric_dtype(series)
                else None
            ),
        })

    return pd.DataFrame(rows)


def detect_target_problems(df):
    """
    Vérifie la structure du target.
    """

    result = {
        "exists": TARGET in df.columns,
        "dtype": None,
        "missing": None,
        "unique_values": [],
        "valid_binary": False,
        "class_0": None,
        "class_1": None,
        "positive_rate": None,
        "unexpected_values": [],
    }

    if TARGET not in df.columns:
        return result

    target = df[TARGET]

    result["dtype"] = str(target.dtype)
    result["missing"] = int(target.isna().sum())

    unique_values = target.dropna().unique()

    # Conversion JSON safe
    result["unique_values"] = [
        value.item()
        if hasattr(value, "item")
        else value
        for value in unique_values
    ]

    valid_values = set(unique_values)

    if valid_values.issubset({0, 1}):
        result["valid_binary"] = True

    else:
        result["unexpected_values"] = [
            value.item()
            if hasattr(value, "item")
            else value
            for value in unique_values
            if value not in {0, 1}
        ]

    result["class_0"] = int(
        (target == 0).sum()
    )

    result["class_1"] = int(
        (target == 1).sum()
    )

    non_missing = target.notna().sum()

    if non_missing > 0:
        result["positive_rate"] = round(
            float(
                (target == 1).sum()
                / non_missing
            ),
            6
        )

    return result


def detect_duplicate_match_ids(df):
    """
    Vérifie les match_id dupliqués.
    """

    if MATCH_ID not in df.columns:
        return pd.DataFrame()

    counts = (
        df[MATCH_ID]
        .value_counts(dropna=False)
        .reset_index()
    )

    counts.columns = [
        MATCH_ID,
        "count"
    ]

    duplicates = counts[
        counts["count"] > 1
    ].copy()

    return duplicates


def inspect_live87_columns(df):
    """
    Identifie les colonnes liées à l'état live 87'.
    """

    live_keywords = [
        "87",
        "live",
        "minute",
        "shot",
        "shots",
        "corner",
        "corners",
        "red",
        "yellow",
        "substitution",
        "substitutions",
        "xg",
        "goal_state",
        "score",
    ]

    columns = []

    for column in df.columns:
        lower = column.lower()

        if any(
            keyword in lower
            for keyword in live_keywords
        ):
            columns.append(column)

    return columns


def check_numeric_anomalies(df):
    """
    Détecte quelques anomalies numériques évidentes.

    Ce n'est pas un contrôle métier complet.
    """

    anomalies = []

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        series = df[column]

        if series.dropna().empty:
            continue

        # Infini
        inf_count = int(
            np.isinf(
                series.astype(float)
            ).sum()
        )

        if inf_count > 0:
            anomalies.append({
                "column": column,
                "issue": "infinite_values",
                "count": inf_count,
            })

        # Valeurs extrêmes via IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            continue

        lower = q1 - 5 * iqr
        upper = q3 + 5 * iqr

        extreme_count = int(
            (
                (series < lower)
                | (series > upper)
            ).sum()
        )

        if extreme_count > 0:
            anomalies.append({
                "column": column,
                "issue": "extreme_values_iqr_5x",
                "count": extreme_count,
            })

    return pd.DataFrame(anomalies)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print_section(
        "DATASET VERIFICATION"
    )

    print(
        f"Dataset : {DATASET_PATH}"
    )

    if not DATASET_PATH.exists():
        print(
            "\n❌ ERREUR : dataset introuvable."
        )

        print(
            f"Place {DATASET_PATH.name} "
            "dans le même dossier que verify_dataset.py."
        )

        return

    # ========================================================
    # CHARGEMENT
    # ========================================================

    print_section("1. CHARGEMENT")

    try:
        df = pd.read_csv(
            DATASET_PATH,
            low_memory=False
        )

    except Exception as e:
        print(
            f"❌ Impossible de charger le dataset : {e}"
        )
        return

    print(
        f"Rows      : {df.shape[0]:,}"
    )

    print(
        f"Columns   : {df.shape[1]:,}"
    )

    # ========================================================
    # STRUCTURE
    # ========================================================

    print_section("2. STRUCTURE")

    structure_ok = True

    if EXPECTED_ROWS:
        if len(df) != EXPECTED_ROWS:
            print(
                f"⚠️ Nombre de lignes différent "
                f"de l'attendu : "
                f"{len(df):,} "
                f"(attendu {EXPECTED_ROWS:,})"
            )

            structure_ok = False

        else:
            print(
                f"✅ Nombre de lignes : {len(df):,}"
            )

    if EXPECTED_COLUMNS:
        if len(df.columns) != EXPECTED_COLUMNS:
            print(
                f"⚠️ Nombre de colonnes différent "
                f"de l'attendu : "
                f"{len(df.columns)} "
                f"(attendu {EXPECTED_COLUMNS})"
            )

            structure_ok = False

        else:
            print(
                f"✅ Nombre de colonnes : "
                f"{len(df.columns)}"
            )

    print(
        f"Memory usage : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )

    # ========================================================
    # TARGET
    # ========================================================

    print_section("3. TARGET")

    target_info = detect_target_problems(df)

    if not target_info["exists"]:

        print(
            f"❌ Target '{TARGET}' introuvable."
        )

    else:

        print(
            f"Target : {TARGET}"
        )

        print(
            f"Dtype : {target_info['dtype']}"
        )

        print(
            f"Missing : {target_info['missing']:,}"
        )

        print(
            f"Unique values : "
            f"{target_info['unique_values']}"
        )

        print(
            f"Class 0 : "
            f"{target_info['class_0']:,}"
        )

        print(
            f"Class 1 : "
            f"{target_info['class_1']:,}"
        )

        if target_info["positive_rate"] is not None:
            print(
                f"Positive rate : "
                f"{target_info['positive_rate'] * 100:.2f}%"
            )

        if target_info["valid_binary"]:
            print(
                "✅ Target binaire valide : {0,1}"
            )
        else:
            print(
                "❌ Target non binaire ou "
                "valeurs inattendues."
            )

    # ========================================================
    # MATCH ID
    # ========================================================

    print_section("4. MATCH ID")

    match_id_info = {
        "exists": MATCH_ID in df.columns,
        "missing": None,
        "unique": None,
        "duplicates": None,
        "unique_ratio": None,
    }

    duplicate_match_ids = pd.DataFrame()

    if MATCH_ID not in df.columns:

        print(
            f"⚠️ Colonne '{MATCH_ID}' introuvable."
        )

    else:

        match_id = df[MATCH_ID]

        match_id_info["missing"] = int(
            match_id.isna().sum()
        )

        match_id_info["unique"] = int(
            match_id.nunique(dropna=True)
        )

        match_id_info["duplicates"] = int(
            match_id.duplicated(
                keep=False
            ).sum()
        )

        match_id_info["unique_ratio"] = round(
            float(
                match_id.nunique(dropna=True)
                / max(len(df), 1)
            ),
            6
        )

        print(
            f"Missing : "
            f"{match_id_info['missing']:,}"
        )

        print(
            f"Unique : "
            f"{match_id_info['unique']:,}"
        )

        print(
            f"Duplicate rows : "
            f"{match_id_info['duplicates']:,}"
        )

        print(
            f"Unique ratio : "
            f"{match_id_info['unique_ratio'] * 100:.2f}%"
        )

        duplicate_match_ids = (
            detect_duplicate_match_ids(df)
        )

        if duplicate_match_ids.empty:
            print(
                "✅ Aucun match_id dupliqué."
            )

        else:
            print(
                f"⚠️ "
                f"{len(duplicate_match_ids):,} "
                "match_id(s) apparaissent plusieurs fois."
            )

            duplicate_match_ids.to_csv(
                OUTPUT_DIR / "duplicate_match_ids.csv",
                index=False
            )

    # ========================================================
    # VALEURS MANQUANTES
    # ========================================================

    print_section("5. VALEURS MANQUANTES")

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing_pct = (
        missing / len(df) * 100
    )

    missing_df = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": missing_pct.values,
    })

    missing_df = missing_df[
        missing_df["missing_count"] > 0
    ]

    if missing_df.empty:

        print(
            "✅ Aucune valeur manquante."
        )

    else:

        print(
            f"⚠️ Colonnes avec valeurs manquantes : "
            f"{len(missing_df)}"
        )

        print()

        print(
            missing_df.head(20).to_string(
                index=False
            )
        )

        missing_df.to_csv(
            OUTPUT_DIR / "missing_values.csv",
            index=False
        )

    # ========================================================
    # TYPES DE FEATURES
    # ========================================================

    print_section("6. TYPES DE FEATURES")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    # Retirer target des features
    numeric_features = [
        c for c in numeric_columns
        if c != TARGET
    ]

    categorical_features = [
        c for c in categorical_columns
        if c != TARGET
    ]

    print(
        f"Numerical columns : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical/object columns : "
        f"{len(categorical_features)}"
    )

    if categorical_features:
        print(
            "\nCategorical columns :"
        )

        for column in categorical_features:
            print(
                f"  - {column}"
            )

    # ========================================================
    # DATES
    # ========================================================

    print_section("7. DETECTION DES DATES")

    date_columns = detect_date_columns(df)

    date_results = []

    if not date_columns:

        print(
            "⚠️ Aucune colonne de date détectée."
        )

        print(
            "\nIMPORTANT :"
        )

        print(
            "La validation chronologique actuelle "
            "dépend donc de l'ordre des lignes du CSV."
        )

        print(
            "\nIl faut vérifier que le CSV est bien "
            "trié du plus ancien au plus récent."
        )

    else:

        print(
            f"Colonnes de date candidates : "
            f"{len(date_columns)}"
        )

        for column in date_columns:

            print(
                f"\nAnalyse : {column}"
            )

            result = analyze_date_column(
                df,
                column
            )

            date_results.append(result)

            print(
                f"  Parse success : "
                f"{result['parse_success_rate']}"
            )

            print(
                f"  Min : "
                f"{result['min_date']}"
            )

            print(
                f"  Max : "
                f"{result['max_date']}"
            )

            print(
                f"  Order : "
                f"{result['chronological_order']}"
            )

    # ========================================================
    # ORDRE DES LIGNES SANS DATE
    # ========================================================

    print_section(
        "8. VERIFICATION DE L'ORDRE DES LIGNES"
    )

    row_order_info = {
        "date_available": bool(date_columns),
        "row_order_status": "UNKNOWN",
    }

    if date_columns:

        increasing = [
            result
            for result in date_results
            if result["chronological_order"]
            == "OLDEST_TO_NEWEST"
        ]

        decreasing = [
            result
            for result in date_results
            if result["chronological_order"]
            == "NEWEST_TO_OLDEST"
        ]

        if increasing:

            row_order_info[
                "row_order_status"
            ] = "OLDEST_TO_NEWEST"

            print(
                "✅ Une colonne indique un ordre "
                "ancien → récent."
            )

        elif decreasing:

            row_order_info[
                "row_order_status"
            ] = "NEWEST_TO_OLDEST"

            print(
                "⚠️ Les données semblent être "
                "récentes → anciennes."
            )

        else:

            row_order_info[
                "row_order_status"
            ] = "NOT_SORTED"

            print(
                "❌ Les données ne semblent pas "
                "être triées chronologiquement."
            )

    else:

        print(
            "⚠️ Impossible de confirmer "
            "l'ordre chronologique automatiquement."
        )

        print(
            "Le dataset doit être vérifié "
            "par rapport à sa source."
        )

    # ========================================================
    # LIVE 87
    # ========================================================

    print_section(
        "9. VARIABLES LIVE / 87E MINUTE"
    )

    live87_columns = inspect_live87_columns(
        df
    )

    print(
        f"Colonnes potentiellement liées "
        f"au Live-87 : {len(live87_columns)}"
    )

    if live87_columns:

        for column in live87_columns:
            print(
                f"  - {column}"
            )

    else:

        print(
            "⚠️ Aucune colonne Live-87 "
            "évidente détectée."
        )

    # ========================================================
    # COLONNES SUSPECTES / LEAKAGE
    # ========================================================

    print_section(
        "10. DETECTION DE COLONNES SUSPECTES"
    )

    suspicious_df = detect_suspicious_columns(
        df
    )

    if suspicious_df.empty:

        print(
            "✅ Aucune colonne suspecte détectée "
            "par les règles automatiques."
        )

    else:

        print(
            f"⚠️ Colonnes suspectes : "
            f"{len(suspicious_df)}"
        )

        print()

        print(
            suspicious_df.to_string(
                index=False
            )
        )

        suspicious_df.to_csv(
            OUTPUT_DIR / "suspicious_columns.csv",
            index=False
        )

    # ========================================================
    # ANOMALIES NUMERIQUES
    # ========================================================

    print_section(
        "11. ANOMALIES NUMERIQUES"
    )

    anomaly_df = check_numeric_anomalies(
        df
    )

    if anomaly_df.empty:

        print(
            "✅ Aucune anomalie numérique "
            "évidente détectée."
        )

    else:

        print(
            f"⚠️ Anomalies détectées : "
            f"{len(anomaly_df)}"
        )

        print(
            anomaly_df.head(30).to_string(
                index=False
            )
        )

        anomaly_df.to_csv(
            OUTPUT_DIR / "numeric_anomalies.csv",
            index=False
        )

    # ========================================================
    # PROFIL DES COLONNES
    # ========================================================

    print_section(
        "12. PROFIL COMPLET DES COLONNES"
    )

    column_profile = build_column_profile(
        df
    )

    column_profile.to_csv(
        OUTPUT_DIR / "column_profile.csv",
        index=False
    )

    print(
        "✅ column_profile.csv créé."
    )

    # ========================================================
    # CORRELATION TARGET
    # ========================================================

    print_section(
        "13. CORRELATION AVEC LE TARGET"
    )

    target_correlations = []

    if TARGET in df.columns:

        numeric_df = df.select_dtypes(
            include=np.number
        )

        for column in numeric_df.columns:

            if column == TARGET:
                continue

            try:

                correlation = (
                    numeric_df[column]
                    .corr(
                        numeric_df[TARGET]
                    )
                )

                if pd.notna(correlation):

                    target_correlations.append({
                        "column": column,
                        "correlation": float(
                            correlation
                        ),
                        "abs_correlation": abs(
                            float(correlation)
                        ),
                    })

            except Exception:
                continue

    correlation_df = pd.DataFrame(
        target_correlations
    )

    if not correlation_df.empty:

        correlation_df = (
            correlation_df
            .sort_values(
                "abs_correlation",
                ascending=False
            )
        )

        print(
            correlation_df.head(20).to_string(
                index=False
            )
        )

        correlation_df.to_csv(
            OUTPUT_DIR /
            "target_correlations.csv",
            index=False
        )

    # ========================================================
    # FIRST / LAST ROWS
    # ========================================================

    print_section(
        "14. INSPECTION PREMIERE / DERNIERE LIGNES"
    )

    print("\n--- PREMIERES LIGNES ---")

    print(
        df.head(5).to_string(
            index=False
        )
    )

    print("\n--- DERNIERES LIGNES ---")

    print(
        df.tail(5).to_string(
            index=False
        )
    )

    # ========================================================
    # STABILITE DU TARGET PAR BLOCS
    # ========================================================

    print_section(
        "15. STABILITE DU TARGET PAR BLOCS"
    )

    block_results = []

    number_of_blocks = 10

    block_size = len(df) // number_of_blocks

    if TARGET in df.columns:

        for i in range(number_of_blocks):

            start = i * block_size

            if i == number_of_blocks - 1:
                end = len(df)

            else:
                end = (i + 1) * block_size

            block = df.iloc[
                start:end
            ]

            rate = block[TARGET].mean()

            block_results.append({
                "block": i + 1,
                "start_row": start,
                "end_row": end - 1,
                "rows": len(block),
                "goal_rate": float(rate),
            })

            print(
                f"Block {i + 1:02d} | "
                f"rows {start:,}-{end - 1:,} | "
                f"goal rate = {rate * 100:.2f}%"
            )

    block_df = pd.DataFrame(
        block_results
    )

    if not block_df.empty:

        block_df.to_csv(
            OUTPUT_DIR /
            "target_by_blocks.csv",
            index=False
        )

    # ========================================================
    # VERDICT
    # ========================================================

    print_section(
        "16. VERDICT FINAL"
    )

    errors = []
    warnings_list = []
    positives = []

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    if not target_info["exists"]:

        errors.append(
            f"Target '{TARGET}' absent."
        )

    elif target_info["missing"] > 0:

        errors.append(
            f"Target contient "
            f"{target_info['missing']} "
            "valeur(s) manquante(s)."
        )

    elif not target_info["valid_binary"]:

        errors.append(
            "Target non binaire."
        )

    else:

        positives.append(
            "Target binaire valide."
        )

    # --------------------------------------------------------
    # Match ID
    # --------------------------------------------------------

    if not match_id_info["exists"]:

        warnings_list.append(
            f"Colonne '{MATCH_ID}' absente."
        )

    elif match_id_info["duplicates"] > 0:

        warnings_list.append(
            f"{match_id_info['duplicates']:,} "
            "lignes ont un match_id dupliqué."
        )

    else:

        positives.append(
            "match_id unique."
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if not date_columns:

        warnings_list.append(
            "Aucune date détectée : "
            "l'ordre chronologique ne peut "
            "pas être confirmé automatiquement."
        )

    elif row_order_info[
        "row_order_status"
    ] == "NOT_SORTED":

        warnings_list.append(
            "Les lignes ne sont pas "
            "triées chronologiquement."
        )

    elif row_order_info[
        "row_order_status"
    ] == "NEWEST_TO_OLDEST":

        warnings_list.append(
            "Les lignes sont triées "
            "récent → ancien."
        )

    elif row_order_info[
        "row_order_status"
    ] == "OLDEST_TO_NEWEST":

        positives.append(
            "Ordre chronologique "
            "ancien → récent détecté."
        )

    # --------------------------------------------------------
    # Missing
    # --------------------------------------------------------

    if not missing_df.empty:

        max_missing_pct = (
            missing_df["missing_pct"].max()
        )

        warnings_list.append(
            f"Valeurs manquantes présentes "
            f"(max {max_missing_pct:.2f}% "
            "sur une colonne)."
        )

    else:

        positives.append(
            "Aucune valeur manquante."
        )

    # --------------------------------------------------------
    # Suspicious
    # --------------------------------------------------------

    if not suspicious_df.empty:

        high_risk = suspicious_df[
            suspicious_df["severity"] == "HIGH"
        ]

        if len(high_risk) > 0:

            warnings_list.append(
                f"{len(high_risk)} colonne(s) "
                "présentent un risque potentiel "
                "de data leakage."
            )

    else:

        positives.append(
            "Aucune colonne suspecte "
            "automatiquement détectée."
        )

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    if errors:

        verdict = "FAIL"

    elif any(
        "data leakage" in warning.lower()
        for warning in warnings_list
    ):

        verdict = "WARNING_HIGH_RISK"

    elif warnings_list:

        verdict = "WARNING"

    else:

        verdict = "PASS"

    print(
        f"\nVERDICT : {verdict}"
    )

    if positives:

        print(
            "\n✅ POINTS POSITIFS"
        )

        for item in positives:
            print(
                f"  ✓ {item}"
            )

    if warnings_list:

        print(
            "\n⚠️ AVERTISSEMENTS"
        )

        for item in warnings_list:
            print(
                f"  ! {item}"
            )

    if errors:

        print(
            "\n❌ ERREURS"
        )

        for item in errors:
            print(
                f"  ✗ {item}"
            )

    # ========================================================
    # RESUME JSON
    # ========================================================

    summary = {
        "dataset": str(DATASET_PATH),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "expected_rows": EXPECTED_ROWS,
        "expected_columns": EXPECTED_COLUMNS,
        "structure_ok": structure_ok,
        "target": target_info,
        "match_id": match_id_info,
        "date_columns": date_columns,
        "date_analysis": date_results,
        "row_order": row_order_info,
        "numeric_features": len(
            numeric_features
        ),
        "categorical_features": len(
            categorical_features
        ),
        "live87_columns": live87_columns,
        "missing_columns": int(
            len(missing_df)
        ),
        "suspicious_columns": int(
            len(suspicious_df)
        ),
        "numeric_anomalies": int(
            len(anomaly_df)
        ),
        "verdict": verdict,
        "positives": positives,
        "warnings": warnings_list,
        "errors": errors,
    }

    with open(
        OUTPUT_DIR /
        "verification_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # RAPPORT TXT
    # ========================================================

    report_path = (
        OUTPUT_DIR /
        "verification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "DATASET VERIFICATION REPORT\n"
        )

        report.write(
            "=" * 70 + "\n\n"
        )

        report.write(
            f"Dataset: {DATASET_PATH}\n"
        )

        report.write(
            f"Rows: {df.shape[0]:,}\n"
        )

        report.write(
            f"Columns: {df.shape[1]:,}\n\n"
        )

        report.write(
            f"VERDICT: {verdict}\n\n"
        )

        report.write(
            "TARGET\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            json.dumps(
                target_info,
                indent=2,
                ensure_ascii=False
            )
        )

        report.write(
            "\n\nMATCH ID\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            json.dumps(
                match_id_info,
                indent=2,
                ensure_ascii=False
            )
        )

        report.write(
            "\n\nDATE / CHRONOLOGY\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        report.write(
            json.dumps(
                {
                    "columns": date_columns,
                    "analysis": date_results,
                    "row_order": row_order_info,
                },
                indent=2,
                ensure_ascii=False
            )
        )

        report.write(
            "\n\nLIVE-87 COLUMNS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        for column in live87_columns:

            report.write(
                f"- {column}\n"
            )

        report.write(
            "\n\nWARNINGS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        for warning in warnings_list:

            report.write(
                f"- {warning}\n"
            )

        report.write(
            "\n\nERRORS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        for error in errors:

            report.write(
                f"- {error}\n"
            )

        report.write(
            "\n\nPOSITIVE CHECKS\n"
        )

        report.write(
            "-" * 40 + "\n"
        )

        for positive in positives:

            report.write(
                f"- {positive}\n"
            )

    # ========================================================
    # FIN
    # ========================================================

    print_section(
        "17. FICHIERS GENERES"
    )

    print(
        f"📁 {OUTPUT_DIR}"
    )

    print(
        "  - verification_report.txt"
    )

    print(
        "  - verification_summary.json"
    )

    print(
        "  - column_profile.csv"
    )

    if not missing_df.empty:
        print(
            "  - missing_values.csv"
        )

    if not duplicate_match_ids.empty:
        print(
            "  - duplicate_match_ids.csv"
        )

    if not suspicious_df.empty:
        print(
            "  - suspicious_columns.csv"
        )

    if not anomaly_df.empty:
        print(
            "  - numeric_anomalies.csv"
        )

    if not correlation_df.empty:
        print(
            "  - target_correlations.csv"
        )

    if not block_df.empty:
        print(
            "  - target_by_blocks.csv"
        )

    print(
        "\n✅ Vérification terminée."
    )

    print(
        "\nIMPORTANT : "
        "le script ne modifie pas le dataset original."
    )


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    main()