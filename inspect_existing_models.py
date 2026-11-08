from pathlib import Path
import joblib
import json
import traceback

PROJECT_DIR = Path("/Users/diakiteseyd/Desktop/ML_Goal_After_87:")

MODEL_PATHS = [
    PROJECT_DIR / "outputs/goal_after_87_random_forest.joblib",
    PROJECT_DIR / "outputs/73_27/goal_after_87_random_forest.joblib",
    PROJECT_DIR / "outputs/67_33/goal_after_87_random_forest.joblib",
    PROJECT_DIR / "outputs/50_50/goal_after_87_random_forest.joblib",
]


def inspect_object(obj, indent=""):
    print(f"{indent}Type : {type(obj)}")

    # Attributs généraux
    attrs = [
        "n_features_in_",
        "feature_names_in_",
        "classes_",
        "n_estimators",
        "max_depth",
        "criterion",
        "random_state",
    ]

    for attr in attrs:
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)

                if attr == "feature_names_in_":
                    print(f"{indent}{attr} :")
                    for i, feature in enumerate(value):
                        print(f"{indent}  [{i:02d}] {feature}")
                else:
                    print(f"{indent}{attr} : {value}")
            except Exception as e:
                print(f"{indent}{attr} : <erreur: {e}>")

    # Pipeline
    if hasattr(obj, "steps"):
        print()
        print(f"{indent}PIPELINE")
        print(f"{indent}" + "-" * 50)

        for name, step in obj.steps:
            print(f"{indent}Étape : {name}")
            print(f"{indent}  Type : {type(step)}")

            if hasattr(step, "n_features_in_"):
                print(f"{indent}  n_features_in_ : {step.n_features_in_}")

            if hasattr(step, "feature_names_in_"):
                print(f"{indent}  feature_names_in_ :")
                for i, feature in enumerate(step.feature_names_in_):
                    print(f"{indent}    [{i:02d}] {feature}")

            # ColumnTransformer
            if hasattr(step, "transformers_"):
                print(f"{indent}  ColumnTransformer détecté")

                try:
                    for transformer_name, transformer, columns in step.transformers_:
                        print(
                            f"{indent}    - {transformer_name}: "
                            f"{type(transformer)}"
                        )
                        print(f"{indent}      Colonnes : {columns}")
                except Exception as e:
                    print(
                        f"{indent}    Impossible de lire transformers_: {e}"
                    )

            # RandomForest / ExtraTrees / GradientBoosting etc.
            if hasattr(step, "feature_importances_"):
                try:
                    importances = step.feature_importances_

                    print(f"{indent}  Feature importances disponibles")
                    print(
                        f"{indent}  Nombre d'importances : "
                        f"{len(importances)}"
                    )

                    if hasattr(step, "feature_names_in_"):
                        names = step.feature_names_in_

                        pairs = list(zip(names, importances))
                        pairs.sort(key=lambda x: x[1], reverse=True)

                        print(f"{indent}  Top 20 features :")

                        for rank, (name, importance) in enumerate(
                            pairs[:20], 1
                        ):
                            print(
                                f"{indent}    {rank:02d}. "
                                f"{name}: {importance:.8f}"
                            )
                except Exception as e:
                    print(
                        f"{indent}  Erreur feature_importances_: {e}"
                    )

            print()


def inspect_model(path):
    print()
    print("=" * 80)
    print("MODÈLE")
    print("=" * 80)
    print(path)

    if not path.exists():
        print("✗ Fichier introuvable")
        return

    try:
        model = joblib.load(path)

        print()
        print("-" * 80)
        print("STRUCTURE")
        print("-" * 80)

        inspect_object(model)

        # Si pipeline, inspecter explicitement la dernière étape
        if hasattr(model, "steps"):
            print()
            print("=" * 80)
            print("DÉTAIL DES ÉTAPES DU PIPELINE")
            print("=" * 80)

            for name, step in model.steps:
                print()
                print(f"[{name}]")
                print(f"Type : {type(step)}")

                if hasattr(step, "get_feature_names_out"):
                    try:
                        feature_names = step.get_feature_names_out()

                        print(
                            f"Nombre de features transformées : "
                            f"{len(feature_names)}"
                        )

                        print("Features transformées :")

                        for i, feature in enumerate(feature_names):
                            print(f"  [{i:03d}] {feature}")

                    except Exception as e:
                        print(
                            f"get_feature_names_out() impossible : {e}"
                        )

                if hasattr(step, "feature_names_in_"):
                    print("Features d'entrée :")

                    for i, feature in enumerate(
                        step.feature_names_in_
                    ):
                        print(f"  [{i:03d}] {feature}")

        # Feature names directement sur le modèle
        if hasattr(model, "feature_names_in_"):
            print()
            print("=" * 80)
            print("FEATURES DU MODÈLE")
            print("=" * 80)

            features = list(model.feature_names_in_)

            print(f"Nombre total : {len(features)}")

            for i, feature in enumerate(features):
                print(f"{i:03d} : {feature}")

            # Sauvegarde automatique
            output_name = path.stem + "_features.json"
            output_path = path.parent / output_name

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(features, f, indent=2, ensure_ascii=False)

            print()
            print(f"✓ Features sauvegardées : {output_path}")

    except Exception as e:
        print()
        print("✗ ERREUR lors du chargement du modèle")
        print(str(e))
        print()
        traceback.print_exc()


def main():
    print()
    print("=" * 80)
    print("INSPECTION DES MODÈLES ML EXISTANTS")
    print("=" * 80)

    print()
    print(f"Projet : {PROJECT_DIR}")

    existing = []

    for path in MODEL_PATHS:
        if path.exists():
            existing.append(path)

    print()
    print(f"Modèles trouvés : {len(existing)}")

    for i, path in enumerate(existing, 1):
        print(f"  {i}. {path}")

    if not existing:
        print()
        print("Aucun modèle trouvé.")
        return

    for path in existing:
        inspect_model(path)

    print()
    print("=" * 80)
    print("INSPECTION TERMINÉE")
    print("=" * 80)
    print()
    print("IMPORTANT :")
    print(
        "Envoie-moi tout le résultat du terminal. "
        "À partir de celui-ci, nous déterminerons exactement "
        "quel modèle et quelles features utiliser pour le scanner live."
    )


if __name__ == "__main__":
    main()