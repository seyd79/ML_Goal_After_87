import os
from pathlib import Path

ROOT = Path(".").resolve()

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", ".pytest_cache",
    ".mypy_cache", ".idea"
}

DATA_EXTENSIONS = {
    ".csv", ".json", ".jsonl",
    ".parquet", ".xlsx", ".xlsm", ".txt"
}

STRONG_TERMS = (
    "test_futur",
    "test-futur",
    "protected",
    "protege",
    "holdout",
    "golden_test",
    "locked_test",
)

SOURCE_TERMS = (
    "TEST_FUTUR",
    "protected_ids",
    "protected_sets",
    "PROTECTED",
    "HOLDOUT",
    "holdout",
    "test_futur",
)

MAX_PRINT = 100

data_files = []
code_references = []
scanned_py = 0

print("=" * 72)
print("ETAPE 389 - INVENTAIRE DES ENSEMBLES PROTEGES")
print("=" * 72)

for current, dirs, files in os.walk(ROOT):
    dirs[:] = sorted(
        d for d in dirs
        if d not in SKIP_DIRS and not d.startswith(".")
    )

    folder = Path(current)

    for name in sorted(files):
        path = folder / name
        relative = path.relative_to(ROOT).as_posix()

        if (
            path.suffix.lower() in DATA_EXTENSIONS
            and any(term in name.lower() for term in STRONG_TERMS)
        ):
            data_files.append(relative)

        if path.suffix.lower() != ".py":
            continue

        scanned_py += 1

        try:
            # Inspection du code source seulement.
            # Aucun CSV/JSON de matchs n'est ouvert.
            with path.open(
                encoding="utf-8",
                errors="replace"
            ) as f:
                for line_no, line in enumerate(f, 1):
                    if any(term in line for term in SOURCE_TERMS):
                        code_references.append((
                            relative,
                            line_no,
                            line.strip()[:140]
                        ))
        except OSError:
            print("SOURCE INACCESSIBLE :", relative)

print("\n=== FICHIERS POTENTIELLEMENT PROTEGES ===")
print("Nombre de fichiers trouves :", len(data_files))

for relative in data_files[:MAX_PRINT]:
    print(relative)

if len(data_files) > MAX_PRINT:
    print("... autres fichiers :", len(data_files) - MAX_PRINT)

print("\n=== REFERENCES DANS LE CODE ===")
print("Scripts Python parcourus :", scanned_py)
print("References trouvees :", len(code_references))

for path, line_no, content in code_references[:MAX_PRINT]:
    print(f"{path}:{line_no}")
    print("  ", content)

if len(code_references) > MAX_PRINT:
    print(
        "... autres references :",
        len(code_references) - MAX_PRINT
    )

print("\n=== STATUT ===")
print("Inventaire des noms et du code source : TERMINE")
print("Contenu des fichiers de donnees : NON LU")
print("Labels de validation : NON CONSULTES")
print("Predictions V4.1-A2 : INCHANGEES")
print("Correspondance des IDs proteges : NON ENCORE VERIFIEE")
print("Certification complete : NON")
print("Aucun fichier modifie")
print("Aucun appel API")
print("ETAPE 389 TERMINEE")
