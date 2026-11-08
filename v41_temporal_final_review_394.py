import ast
import hashlib
from pathlib import Path

FILES = {
    "v41_live87_own_goal_364.py": [
        ("Parsing des minutes", 875, 990),
        ("Traitement des tirs shotmap", 2030, 2205),
        ("Fallback timeline", 2350, 2450),
    ],
    "v41_prematch_safe_bridge_318.py": [
        ("Branchement du bridge", 60, 180),
    ],
    "collect_historical_dataset_v2.py": [
        ("Configuration de collecte", 1220, 1305),
        ("Assemblage des features", 1315, 1375),
    ],
    "v41_prematch_temporal_adapter_313.py": [
        ("Adaptateur temporel", 95, 160),
    ],
}

OUT = Path("v41_temporal_final_review_394_output.txt")

if OUT.exists():
    raise RuntimeError("STOP : rapport 394 deja present")

output = []
missing = []

def emit(line=""):
    output.append(str(line))

def section(filename, lines, title, start, end):
    emit("=" * 72)
    emit(f"{title} | {filename}:{start}-{end}")
    emit("=" * 72)

    for number in range(start, min(end, len(lines)) + 1):
        emit(f"{number:5d} | {lines[number-1]}")

emit("ETAPE 394 - REVUE TEMPORELLE CIBLEE")
emit("AUDIT STATIQUE SANS DONNEES DE MATCHS")
emit()

for filename, sections in FILES.items():
    path = Path(filename)

    if not path.is_file():
        missing.append(filename)
        continue

    source = path.read_text(
        encoding="utf-8",
        errors="replace"
    )
    ast.parse(source, filename=filename)
    lines = source.splitlines()

    emit()
    emit("SOURCE : " + filename)
    emit("SHA256 : " + hashlib.sha256(
        path.read_bytes()
    ).hexdigest())

    for title, start, end in sections:
        section(filename, lines, title, start, end)

# Examiner aussi le filtre d'origine, sur lequel
# l'adaptateur temporel V4.1 s'appuie.
path = Path("rebuild_clean_prematch_v2.py")

if not path.is_file():
    missing.append(str(path))
else:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()

    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.name == "filter_before_cutoff"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "STOP : filtre source absent ou ambigu"
        )

    node = matches[0]

    section(
        str(path),
        lines,
        "Filtre historique original",
        node.lineno,
        node.end_lineno,
    )

if missing:
    raise RuntimeError(
        "STOP : fichiers manquants : "
        + ", ".join(missing)
    )

emit()
emit("=" * 72)
emit("POINTS A ANALYSER")
emit("=" * 72)
emit("1. Minutes 87+1, 88, 90+2")
emit("2. Traitement shotmap et fallback")
emit("3. Utilisation effective du bridge V4.1")
emit("4. Exclusion des historiques futurs")
emit("5. Disponibilite reelle en direct : non certifiee")
emit()
emit("Aucun label ni resultat de validation consulte.")
emit("Aucun appel API.")
emit("Aucun modele entraine.")
emit("Predictions et seuils inchanges.")
emit("ETAPE 394 TERMINEE.")

OUT.write_text(
    "\n".join(output) + "\n",
    encoding="utf-8"
)

print("=" * 72)
print("ETAPE 394 - RAPPORT GENERE")
print("=" * 72)
print("Fichier :", OUT.resolve())
print("Lignes :", len(output))
print("Labels de validation : NON CONSULTES")
print("Predictions : INCHANGEES")
print("ETAPE 394 TERMINEE")
