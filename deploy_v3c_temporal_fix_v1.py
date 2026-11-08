import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LIVE = Path("live_scanner_engine.py")
CANDIDATE = Path(
    "live_scanner_engine_temporal_fix_candidate.py"
)
BACKUP = Path(
    "backups/temporal_fix_v1/"
    "live_scanner_engine_before_fix_20261028_175042.py"
)

EXPECTED_ORIGINAL = (
    "04365d39beaad48d115cbc70d23614f78806eb5e14bc77c08173cc0a29796961"
)
EXPECTED_CANDIDATE = (
    "bffeab39c076406f7470aa3cd6bd8de1811e0a929b75891d5e2fa838d4831719"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_processes():
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )

    matches = []

    for line in result.stdout.splitlines():
        if "live_scanner_engine.py" in line:
            if "deploy_v3c_temporal_fix_v1.py" not in line:
                matches.append(line.strip())

    return matches


parser = argparse.ArgumentParser()
parser.add_argument(
    "--apply",
    action="store_true",
    help="Appliquer le correctif après vérifications"
)
args = parser.parse_args()

print("=== ÉTAPE 156 : DÉPLOIEMENT CONTRÔLÉ ===")

for path in (LIVE, CANDIDATE, BACKUP):
    if not path.is_file():
        raise SystemExit(f"FICHIER MANQUANT : {path}")

if sha(LIVE) != EXPECTED_ORIGINAL:
    raise SystemExit(
        "ARRÊT : le LIVE ne correspond plus à "
        "la version originale auditée."
    )

if sha(BACKUP) != EXPECTED_ORIGINAL:
    raise SystemExit(
        "ARRÊT : sauvegarde non conforme."
    )

if sha(CANDIDATE) != EXPECTED_CANDIDATE:
    raise SystemExit(
        "ARRÊT : candidat différent de la version testée."
    )

print("Empreinte LIVE : PASS")
print("Sauvegarde : PASS")
print("Candidat : PASS")

running = check_processes()

if running:
    print("\nPROCESSUS DÉTECTÉS :")
    for process in running:
        print(process)
    raise SystemExit(
        "ARRÊT : arrêter d'abord le scanner concerné."
    )

print("Aucun processus LIVE identifié par son nom.")

if not args.apply:
    print("\nMODE VÉRIFICATION : PASS")
    print("Aucun fichier modifié.")
    print("Pour appliquer : utiliser --apply")
    sys.exit(0)

fd, temp_path = tempfile.mkstemp(
    prefix=".v3c_temporal_fix_",
    suffix=".py",
    dir=str(LIVE.parent),
)
os.close(fd)
temp = Path(temp_path)

try:
    shutil.copy2(CANDIDATE, temp)

    if sha(temp) != EXPECTED_CANDIDATE:
        raise RuntimeError("Copie temporaire non conforme")

    compile(
        temp.read_bytes(),
        str(temp),
        "exec",
    )

    os.replace(temp, LIVE)

    if sha(LIVE) != EXPECTED_CANDIDATE:
        raise RuntimeError("Empreinte déployée incorrecte")

    print("\nDÉPLOIEMENT : PASS")
    print("Correctif temporel installé.")
    print("Modèle et schéma inchangés.")
    print("Scanner non démarré.")

except Exception as exc:
    if temp.exists():
        temp.unlink()

    if sha(LIVE) != EXPECTED_ORIGINAL:
        shutil.copy2(BACKUP, LIVE)

    raise SystemExit(
        f"ÉCHEC, restauration effectuée si nécessaire : {exc}"
    )
