from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/temporal_signal_analysis")

FILES = {
    "top_percentages": OUTPUT_DIR / "temporal_top_percentages.csv",
    "thresholds": OUTPUT_DIR / "temporal_thresholds.csv",
    "probability_bins": OUTPUT_DIR / "temporal_probability_bins.csv",
    "rank_stability": OUTPUT_DIR / "temporal_rank_stability.csv",
}

MODELS = ["50_50", "67_33", "73_27"]

PERCENTILES = [30, 25, 20, 15, 10, 5]


# ============================================================
# UTILITAIRES
# ============================================================

def print_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_column(df, candidates):
    """
    Trouve automatiquement une colonne parmi plusieurs noms possibles.
    """
    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    return None


def load_csv(path):
    if not path.exists():
        print(f"[WARNING] Fichier absent : {path}")
        return None

    try:
        df = pd.read_csv(path)

        print(f"[OK] {path.name}")
        print(f"     lignes : {len(df):,}")
        print(f"     colonnes : {len(df.columns)}")
        print(f"     colonnes détectées : {list(df.columns)}")

        return df

    except Exception as e:
        print(f"[ERROR] Impossible de lire {path}: {e}")
        return None


# ============================================================
# CHARGEMENT
# ============================================================

print_header("CHARGEMENT DES OUTPUTS")

top_df = load_csv(FILES["top_percentages"])
threshold_df = load_csv(FILES["thresholds"])
bins_df = load_csv(FILES["probability_bins"])
stability_df = load_csv(FILES["rank_stability"])


# ============================================================
# 1. ANALYSE TOP PERCENTAGES
# ============================================================

print_header("ANALYSE TOP PERCENTAGES")

top_results = []

if top_df is not None and not top_df.empty:

    print("\nAperçu :")
    print(top_df.head(10).to_string(index=False))

    model_col = find_column(
        top_df,
        ["model", "dataset", "model_name"]
    )

    fold_col = find_column(
        top_df,
        ["fold"]
    )

    percentile_col = find_column(
        top_df,
        [
            "top_percentage",
            "percentage",
            "percentile",
            "top_percent"
        ]
    )

    rate_col = find_column(
        top_df,
        [
            "goal_rate",
            "positive_rate",
            "rate",
            "top_goal_rate"
        ]
    )

    if model_col and percentile_col and rate_col:

        work = top_df.copy()

        work[percentile_col] = pd.to_numeric(
            work[percentile_col],
            errors="coerce"
        )

        work[rate_col] = pd.to_numeric(
            work[rate_col],
            errors="coerce"
        )

        for model in MODELS:

            model_data = work[
                work[model_col].astype(str) == model
            ]

            if model_data.empty:
                continue

            for pct in PERCENTILES:

                pct_data = model_data[
                    np.isclose(
                        model_data[percentile_col],
                        pct
                    )
                ]

                if pct_data.empty:
                    continue

                mean_rate = pct_data[rate_col].mean()
                std_rate = pct_data[rate_col].std()

                if fold_col:
                    fold_values = pct_data[rate_col].dropna()

                    if len(fold_values) > 0:
                        positive_folds = (
                            fold_values > 0
                        ).sum()

                        stable_folds = len(fold_values)
                    else:
                        positive_folds = 0
                        stable_folds = 0
                else:
                    positive_folds = np.nan
                    stable_folds = np.nan

                top_results.append({
                    "model": model,
                    "top_percent": pct,
                    "mean_goal_rate": mean_rate,
                    "std_goal_rate": std_rate,
                    "folds_with_signal": positive_folds,
                    "folds_total": stable_folds,
                })

    else:
        print(
            "\n[WARNING] Impossible d'identifier automatiquement "
            "les colonnes nécessaires dans temporal_top_percentages.csv."
        )


top_summary = pd.DataFrame(top_results)


# ============================================================
# 2. ANALYSE TOP VS BASELINE
# ============================================================

print_header("TOP PERCENTAGES : ANALYSE DETAILLEE")

if not top_summary.empty:

    # Affichage
    display_df = top_summary.copy()

    display_df["mean_goal_rate"] = (
        display_df["mean_goal_rate"] * 100
    )

    display_df["std_goal_rate"] = (
        display_df["std_goal_rate"] * 100
    )

    display_df = display_df.sort_values(
        ["mean_goal_rate"],
        ascending=False
    )

    print(
        display_df.to_string(
            index=False,
            formatters={
                "mean_goal_rate": "{:.2f}%".format,
                "std_goal_rate": "{:.2f}%".format,
            }
        )
    )


# ============================================================
# 3. ANALYSE DES SEUILS
# ============================================================

print_header("ANALYSE DES SEUILS DE PROBABILITE")

threshold_results = []

if threshold_df is not None and not threshold_df.empty:

    print("\nAperçu :")
    print(threshold_df.head(10).to_string(index=False))

    model_col = find_column(
        threshold_df,
        ["model", "dataset", "model_name"]
    )

    threshold_col = find_column(
        threshold_df,
        [
            "threshold",
            "probability_threshold"
        ]
    )

    precision_col = find_column(
        threshold_df,
        ["precision"]
    )

    recall_col = find_column(
        threshold_df,
        ["recall"]
    )

    f1_col = find_column(
        threshold_df,
        ["f1", "f1_score"]
    )

    goal_rate_col = find_column(
        threshold_df,
        [
            "goal_rate",
            "positive_rate",
            "actual_rate"
        ]
    )

    if model_col and threshold_col:

        work = threshold_df.copy()

        work[threshold_col] = pd.to_numeric(
            work[threshold_col],
            errors="coerce"
        )

        for col in [
            precision_col,
            recall_col,
            f1_col,
            goal_rate_col
        ]:
            if col:
                work[col] = pd.to_numeric(
                    work[col],
                    errors="coerce"
                )

        for model in MODELS:

            model_data = work[
                work[model_col].astype(str) == model
            ]

            if model_data.empty:
                continue

            print(f"\n--- MODELE {model} ---")

            cols = [threshold_col]

            if precision_col:
                cols.append(precision_col)

            if recall_col:
                cols.append(recall_col)

            if f1_col:
                cols.append(f1_col)

            if goal_rate_col:
                cols.append(goal_rate_col)

            output = model_data[cols].sort_values(
                threshold_col
            )

            print(output.to_string(index=False))

            # meilleur F1
            if f1_col:
                valid = model_data.dropna(
                    subset=[f1_col]
                )

                if not valid.empty:

                    best = valid.loc[
                        valid[f1_col].idxmax()
                    ]

                    threshold_results.append({
                        "model": model,
                        "best_threshold": best[threshold_col],
                        "best_f1": best[f1_col],
                        "precision": (
                            best[precision_col]
                            if precision_col else np.nan
                        ),
                        "recall": (
                            best[recall_col]
                            if recall_col else np.nan
                        ),
                    })


# ============================================================
# 4. PROBABILITY BINS
# ============================================================

print_header("ANALYSE DES BINS DE PROBABILITE")

if bins_df is not None and not bins_df.empty:

    print("\nAperçu :")
    print(bins_df.head(20).to_string(index=False))

    model_col = find_column(
        bins_df,
        ["model", "dataset", "model_name"]
    )

    rate_col = find_column(
        bins_df,
        [
            "goal_rate",
            "positive_rate",
            "actual_rate"
        ]
    )

    bin_col = find_column(
        bins_df,
        [
            "bin",
            "probability_bin",
            "probability_range"
        ]
    )

    if model_col and rate_col:

        work = bins_df.copy()

        work[rate_col] = pd.to_numeric(
            work[rate_col],
            errors="coerce"
        )

        for model in MODELS:

            model_data = work[
                work[model_col].astype(str) == model
            ]

            if model_data.empty:
                continue

            print(f"\n--- MODELE {model} ---")

            cols = []

            if bin_col:
                cols.append(bin_col)

            cols.append(rate_col)

            print(
                model_data[cols].to_string(
                    index=False
                )
            )


# ============================================================
# 5. RANK STABILITY
# ============================================================

print_header("ANALYSE DE STABILITE DU RANKING")

if stability_df is not None and not stability_df.empty:

    print("\nAperçu :")
    print(
        stability_df.head(20).to_string(
            index=False
        )
    )

    model_col = find_column(
        stability_df,
        ["model", "dataset", "model_name"]
    )

    if model_col:

        for model in MODELS:

            model_data = stability_df[
                stability_df[model_col].astype(str) == model
            ]

            if model_data.empty:
                continue

            print(f"\n--- {model} ---")
            print(
                model_data.to_string(
                    index=False
                )
            )


# ============================================================
# 6. CLASSEMENT DES MODELES
# ============================================================

print_header("CLASSEMENT DES MODELES")

ranking = []

if not top_summary.empty:

    for model in MODELS:

        model_data = top_summary[
            top_summary["model"] == model
        ]

        if model_data.empty:
            continue

        best_row = model_data.loc[
            model_data["mean_goal_rate"].idxmax()
        ]

        # moyenne sur toutes les zones
        mean_rate = model_data[
            "mean_goal_rate"
        ].mean()

        ranking.append({
            "model": model,
            "best_top_percent": best_row["top_percent"],
            "best_goal_rate": best_row["mean_goal_rate"],
            "mean_top_rates": mean_rate,
            "stability": (
                best_row["folds_with_signal"]
                / best_row["folds_total"]
                if best_row["folds_total"] > 0
                else np.nan
            )
        })


ranking_df = pd.DataFrame(ranking)

if not ranking_df.empty:

    ranking_df = ranking_df.sort_values(
        [
            "stability",
            "best_goal_rate"
        ],
        ascending=False
    )

    ranking_display = ranking_df.copy()

    ranking_display["best_goal_rate"] *= 100
    ranking_display["mean_top_rates"] *= 100
    ranking_display["stability"] *= 100

    print(
        ranking_display.to_string(
            index=False,
            formatters={
                "best_goal_rate": "{:.2f}%".format,
                "mean_top_rates": "{:.2f}%".format,
                "stability": "{:.0f}%".format,
            }
        )
    )


# ============================================================
# 7. IDENTIFICATION DU MEILLEUR ZONE
# ============================================================

print_header("RECHERCHE DE LA ZONE DE SIGNAL")

best_candidates = []

if not top_summary.empty:

    for _, row in top_summary.iterrows():

        stability = (
            row["folds_with_signal"]
            / row["folds_total"]
            if row["folds_total"] > 0
            else 0
        )

        # Score volontairement conservateur :
        #
        # - priorité à la stabilité
        # - ensuite taux de buts
        #
        score = (
            stability * 0.60
            +
            row["mean_goal_rate"] * 0.40
        )

        best_candidates.append({
            "model": row["model"],
            "top_percent": row["top_percent"],
            "mean_goal_rate": row["mean_goal_rate"],
            "std_goal_rate": row["std_goal_rate"],
            "stability": stability,
            "score": score,
        })


if best_candidates:

    candidates_df = pd.DataFrame(
        best_candidates
    )

    candidates_df = candidates_df.sort_values(
        "score",
        ascending=False
    )

    print(
        candidates_df.head(15).to_string(
            index=False,
            formatters={
                "mean_goal_rate": "{:.4f}".format,
                "std_goal_rate": "{:.4f}".format,
                "stability": "{:.2%}".format,
                "score": "{:.4f}".format,
            }
        )
    )

else:

    candidates_df = pd.DataFrame()


# ============================================================
# 8. RECOMMANDATION
# ============================================================

print_header("RECOMMANDATION")

if not candidates_df.empty:

    best = candidates_df.iloc[0]

    print(
        f"""
MODELE RECOMMANDE POUR L'ANALYSE :
    {best["model"]}

ZONE :
    Top {best["top_percent"]:.0f}%

TAUX MOYEN DE BUT APRES 87' :
    {best["mean_goal_rate"]:.2%}

STABILITE :
    {best["stability"]:.2%}

SCORE DE SELECTION :
    {best["score"]:.4f}
"""
    )

    print(
        "IMPORTANT : cette recommandation identifie une zone "
        "de signal, pas encore une stratégie de pari rentable."
    )

else:

    print(
        "Impossible d'identifier automatiquement une zone."
    )


# ============================================================
# 9. EXPORT GLOBAL
# ============================================================

summary_path = OUTPUT_DIR / "temporal_signal_summary.csv"

export_parts = []

if not top_summary.empty:

    export_top = top_summary.copy()
    export_top["analysis"] = "top_percentages"

    export_parts.append(export_top)

if not ranking_df.empty:

    export_ranking = ranking_df.copy()
    export_ranking["analysis"] = "model_ranking"

    export_parts.append(export_ranking)

if best_candidates:

    export_candidates = candidates_df.copy()
    export_candidates["analysis"] = "signal_candidates"

    export_parts.append(export_candidates)


if export_parts:

    # Harmonisation des colonnes
    summary = pd.concat(
        export_parts,
        ignore_index=True,
        sort=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("=" * 80)
    print("EXPORT TERMINE")
    print("=" * 80)

    print(
        f"Fichier créé : {summary_path}"
    )


# ============================================================
# FIN
# ============================================================

print()
print("=" * 80)
print("ANALYSE TERMINEE")
print("=" * 80)

print(
    """
Les fichiers originaux n'ont pas été modifiés.

Le script a uniquement :
- lu les outputs existants ;
- analysé les modèles ;
- analysé les percentiles ;
- analysé les seuils ;
- analysé les bins ;
- analysé la stabilité ;
- créé temporal_signal_summary.csv.
"""
)