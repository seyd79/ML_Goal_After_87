import ast
import csv
import inspect
from pathlib import Path

import rebuild_clean_prematch_v2 as prematch
import collect_historical_dataset_v2 as collector

from v41_prematch_temporal_adapter_313 import (
    build_isolated_adapter,
)

MANIFEST = Path(
    "outputs/v4_research/v41_pilot_plan_298/"
    "PILOT12_METADATA_ONLY.csv"
)

class ResearchPrematchProxy:
    """
    Proxy reserve au pipeline de recherche V4.1.

    Toutes les fonctions sont deleguees au module
    original, sauf build_prematch_asof.
    """

    def __init__(self, original_module):
        self._original = original_module
        self.build_prematch_asof = (
            build_isolated_adapter()
        )

    def __getattr__(self, name):
        return getattr(self._original, name)


def check_collector_call():
    """
    Confirme que collect_one_match appelle
    le constructeur via l'argument prematch.
    """
    source_path = Path(
        "collect_historical_dataset_v2.py"
    )

    tree = ast.parse(
        source_path.read_text(encoding="utf-8")
    )

    function = next(
        (
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "collect_one_match"
        ),
        None
    )

    if function is None:
        raise RuntimeError(
            "collect_one_match introuvable"
        )

    calls = []

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue

        fn = node.func

        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "build_prematch_asof"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "prematch"
        ):
            calls.append(node)

    if len(calls) != 1:
        raise RuntimeError(
            "Appel PRE-MATCH inattendu : "
            + str(len(calls))
        )

    return calls[0].lineno


def main():
    print("=" * 76)
    print("ETAPE 314 - BRANCHEMENT ISOLE V4.1")
    print("=" * 76)

    if not MANIFEST.is_file():
        raise RuntimeError(
            "Manifeste pilote introuvable"
        )

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 12:
        raise RuntimeError(
            "Population pilote inattendue"
        )

    ids = [r["match_id"] for r in rows]

    if len(set(ids)) != 12:
        raise RuntimeError(
            "Identifiants invalides"
        )

    if any(
        r.get("collection_authorized") != "NO"
        or r.get("protected_clearance") != "NOT_VERIFIED"
        for r in rows
    ):
        raise RuntimeError(
            "Statuts du manifeste inattendus"
        )

    print("\n=== MANIFESTE ===")
    print("Candidats :", len(rows))
    print(
        "Mois distincts :",
        len({
            (r["competition_id"], r["month"])
            for r in rows
        })
    )
    print("Collecte autorisee : NON")

    print("\n=== APPEL DU COLLECTEUR ===")

    call_line = check_collector_call()

    print(
        "collect_one_match appelle "
        "prematch.build_prematch_asof"
    )
    print("Ligne :", call_line)

    print("\n=== ADAPTATEUR ===")

    original = prematch.build_prematch_asof
    proxy = ResearchPrematchProxy(prematch)
    isolated = proxy.build_prematch_asof

    assert isolated is not original

    assert (
        inspect.signature(isolated)
        == inspect.signature(original)
    )

    assert (
        proxy.filter_before_cutoff
        is prematch.filter_before_cutoff
    )

    assert (
        isolated.__globals__[
            "filter_before_cutoff"
        ]
        is not prematch.filter_before_cutoff
    )

    assert (
        original.__globals__[
            "filter_before_cutoff"
        ]
        is prematch.filter_before_cutoff
    )

    assert proxy.dt_utc_naive is prematch.dt_utc_naive

    print("Signature du constructeur : OK")
    print("Constructeur V4.1 distinct : OK")
    print("Filtre V4.1 dans le constructeur : OK")
    print("Fonctions auxiliaires deleguees : OK")
    print("Fonction originale intacte : OK")

    print("\n=== CONTROLE DU COLLECTEUR ===")

    assert "prematch" in inspect.signature(
        collector.collect_one_match
    ).parameters

    print(
        "Injection possible via argument prematch : OK"
    )

    print("\n=== DECISION ===")
    print("Branchement structurel : PRET")
    print("Execution PRE-MATCH complete : NON TESTEE")
    print("Collecte detaillee : BLOQUEE")
    print("Exclusions protegees : NON CERTIFIEES")

    print("\nAucun appel API.")
    print("Aucune timeline consultee.")
    print("Aucun label consulte.")
    print("Aucun fichier protege ouvert.")
    print("Aucun dataset modifie.")
    print("TEST_FUTUR non consulte.")
    print("V3-C inchange.")
    print("ETAPE 314 TERMINEE.")


if __name__ == "__main__":
    main()
