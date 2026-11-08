from pathlib import Path
import json
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    "/Users/diakiteseyd/Desktop/ML_Goal_After_87:"
)

# Modèle recherché en priorité
MODEL_NAME = "ET_500_d15_73_27.joblib"


# ============================================================
# RECHERCHE DU MODÈLE
# ============================================================

print("=" * 80)
print("RECHERCHE DES ARTIFACTS ML")
print("=" * 80)

print(f"\nProjet : {PROJECT_DIR}")

if not PROJECT_DIR.exists():
    print("\nERREUR : dossier du projet introuvable.")
    raise SystemExit(1)


print("\nRecherche du modèle...")

model_candidates = list(PROJECT_DIR.rglob(MODEL_NAME))

# Si le nom exact n'existe pas, chercher tous les .joblib
if not model_candidates:
    print(
        f"\nAucun fichier nommé '{MODEL_NAME}' trouvé."
    )

    all_joblib = list(PROJECT_DIR.rglob("*.joblib"))

    if all_joblib:
        print("\nFichiers .joblib trouvés :")

        for i, path in enumerate(all_joblib, 1):
            print(f"  {i}. {path}")
    else:
        print("\nAucun fichier .joblib trouvé dans le projet.")

else:
    print("\nModèle trouvé :")

    for path in model_candidates:
        print(f"  {path}")


# ============================================================
# RECHERCHE DES FEATURES
# ============================================================

print("\n" + "=" * 80)
print("RECHERCHE DES FEATURES")
print("=" * 80)

feature_names = [
    "feature_columns.json",
    "features.json",
    "feature_names.json",
    "model_features.json",
    "features_columns.json",
]

feature_files = []

for name in feature_names:
    feature_files.extend(PROJECT_DIR.rglob(name))


# Supprimer les doublons
feature_files = list(dict.fromkeys(feature_files))


if feature_files:

    print("\nFichiers de features trouvés :")

    for i, path in enumerate(feature_files, 1):
        print(f"  {i}. {path}")

else:

    print("\nAucun fichier JSON de features trouvé.")

    print("\nRecherche plus large des fichiers JSON contenant")
    print("'feature' ou 'column' dans le nom...")

    json_candidates = list(PROJECT_DIR.rglob("*.json"))

    relevant_json = [
        path
        for path in json_candidates
        if (
            "feature" in path.name.lower()
            or "column" in path.name.lower()
            or "model" in path.name.lower()
        )
    ]

    if relevant_json:

        print("\nFichiers JSON potentiellement pertinents :")

        for i, path in enumerate(relevant_json, 1):
            print(f"  {i}. {path}")

    else:

        print("\nAucun JSON potentiellement pertinent trouvé.")


# ============================================================
# INSPECTION DU MODÈLE
# ============================================================

if model_candidates:

    model_path = model_candidates[0]

    print("\n" + "=" * 80)
    print("INSPECTION DU MODÈLE")
    print("=" * 80)

    print(f"\nFichier : {model_path}")

    try:

        model = joblib.load(model_path)

        print(f"\nType Python : {type(model)}")

        # Pipeline sklearn
        if hasattr(model, "steps"):

            print("\nPipeline détecté.")

            print("\nÉtapes :")

            for name, step in model.steps:
                print(f"  - {name}: {type(step)}")

        # Nombre de features
        if hasattr(model, "n_features_in_"):

            print(
                f"\nNombre de features attendu : "
                f"{model.n_features_in_}"
            )

        # Feature names
        if hasattr(model, "feature_names_in_"):

            print("\nFEATURES DU MODÈLE :")

            for i, feature in enumerate(
                model.feature_names_in_,
                1
            ):
                print(f"  {i:3}. {feature}")

        # Si pipeline
        if hasattr(model, "named_steps"):

            print("\nInspection des étapes du pipeline :")

            for step_name, step in model.named_steps.items():

                print(
                    f"\n[{step_name}] "
                    f"{type(step).__name__}"
                )

                if hasattr(step, "feature_names_in_"):

                    print(
                        f"Features utilisées : "
                        f"{len(step.feature_names_in_)}"
                    )

                    for i, feature in enumerate(
                        step.feature_names_in_,
                        1
                    ):
                        print(
                            f"  {i:3}. {feature}"
                        )

    except Exception as e:

        print(
            f"\nERREUR lors du chargement du modèle : {e}"
        )


# ============================================================
# INSPECTION DES FICHIERS FEATURES
# ============================================================

if feature_files:

    print("\n" + "=" * 80)
    print("INSPECTION DES FICHIERS FEATURES")
    print("=" * 80)

    for feature_path in feature_files:

        print(
            f"\nFichier : {feature_path}"
        )

        try:

            with open(
                feature_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            print(
                f"Type JSON : {type(data).__name__}"
            )

            if isinstance(data, list):

                print(
                    f"Nombre de features : {len(data)}"
                )

                print("\nFeatures :")

                for i, feature in enumerate(
                    data,
                    1
                ):
                    print(
                        f"  {i:3}. {feature}"
                    )

            elif isinstance(data, dict):

                print(
                    "\nClés principales :"
                )

                for key in data.keys():

                    print(
                        f"  - {key}"
                    )

                # Recherche automatique
                # d'une liste de features
                for key, value in data.items():

                    if isinstance(value, list):

                        print(
                            f"\nListe trouvée dans '{key}'"
                        )

                        print(
                            f"Nombre : {len(value)}"
                        )

                        for i, feature in enumerate(
                            value,
                            1
                        ):
                            print(
                                f"  {i:3}. {feature}"
                            )

        except Exception as e:

            print(
                f"ERREUR lecture JSON : {e}"
            )


# ============================================================
# RÉSUMÉ
# ============================================================

print("\n" + "=" * 80)
print("RÉSUMÉ")
print("=" * 80)

if model_candidates:
    print(
        f"\n✓ Modèle trouvé : "
        f"{model_candidates[0]}"
    )
else:
    print("\n✗ Modèle non trouvé.")

if feature_files:
    print(
        f"✓ Fichier(s) features trouvé(s) : "
        f"{len(feature_files)}"
    )
else:
    print(
        "✗ Fichier feature_columns.json "
        "non trouvé."
    )

print("\nRecherche terminée.")
print("=" * 80)