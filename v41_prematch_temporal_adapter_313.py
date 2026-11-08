import ast
import inspect
import types
from datetime import datetime, timedelta
from pathlib import Path

import rebuild_clean_prematch_v2 as prematch


MARGIN_HOURS = 4

SOURCE = Path("rebuild_clean_prematch_v2.py")


def verify_source_structure():
    """
    Bloque l'adaptateur si les trois appels
    attendus au filtre changent.
    """
    tree = ast.parse(
        SOURCE.read_text(encoding="utf-8")
    )

    target = None

    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "build_prematch_asof"
        ):
            target = node
            break

    if target is None:
        raise RuntimeError(
            "build_prematch_asof introuvable"
        )

    calls = [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "filter_before_cutoff"
    ]

    if len(calls) != 3:
        raise RuntimeError(
            "Structure inattendue : "
            f"{len(calls)} appels au filtre "
            "au lieu de 3"
        )

    print(
        "Appels PRE-MATCH controles :",
        len(calls)
    )

    return True


def filter_before_cutoff_v41(
    lab,
    matches,
    cutoff,
    target_match_id=None,
):
    """
    Filtre V4.1 conservateur.

    Un match historique doit avoir commence
    strictement plus de 4 heures avant le
    kickoff cible.

    Ce n'est pas une preuve d'heure de fin.
    """
    target_dt = prematch.dt_utc_naive(cutoff)

    if target_dt is None:
        raise ValueError(
            "V4.1 : cutoff invalide"
        )

    safe_cutoff = (
        target_dt
        - timedelta(hours=MARGIN_HOURS)
    )

    return prematch.filter_before_cutoff(
        lab,
        matches,
        safe_cutoff,
        target_match_id=target_match_id,
    )


def build_isolated_adapter():
    """
    Cree une copie fonctionnelle isolee.

    Aucun remplacement global du filtre dans
    le module PRE-MATCH d'origine.
    """
    verify_source_structure()

    original = prematch.build_prematch_asof

    if (
        original.__globals__.get(
            "filter_before_cutoff"
        )
        is not prematch.filter_before_cutoff
    ):
        raise RuntimeError(
            "Binding du filtre original inattendu"
        )

    isolated_globals = dict(
        original.__globals__
    )

    isolated_globals[
        "filter_before_cutoff"
    ] = filter_before_cutoff_v41

    adapter = types.FunctionType(
        original.__code__,
        isolated_globals,
        name="build_prematch_asof_v41",
        argdefs=original.__defaults__,
        closure=original.__closure__,
    )

    adapter.__kwdefaults__ = (
        dict(original.__kwdefaults__)
        if original.__kwdefaults__
        else None
    )

    # Recopier les annotations pour conserver
    # la signature inspectable de la fonction.
    adapter.__annotations__ = dict(
        original.__annotations__
    )

    if (
        inspect.signature(adapter)
        != inspect.signature(original)
    ):
        raise RuntimeError(
            "Signature du constructeur modifiee"
        )

    if (
        adapter.__globals__[
            "filter_before_cutoff"
        ]
        is not filter_before_cutoff_v41
    ):
        raise RuntimeError(
            "Filtre V4.1 non branche"
        )

    if (
        prematch.build_prematch_asof
        is not original
        or original.__globals__[
            "filter_before_cutoff"
        ]
        is not prematch.filter_before_cutoff
    ):
        raise RuntimeError(
            "Le module original a ete modifie"
        )

    return adapter


class FakeLab:
    def extract_match_id(self, match):
        return match.get("match_id")

    def extract_match_date(self, match):
        return match.get("utc_date")


def offline_test():
    adapter = build_isolated_adapter()

    target = datetime(
        2026, 5, 10, 18, 0
    )

    history = [
        {
            "match_id": "ancien",
            "utc_date": "2026-05-09T19:00:00Z",
        },
        {
            "match_id": "cinq_heures",
            "utc_date": "2026-05-10T13:00:00Z",
        },
        {
            "match_id": "quatre_heures",
            "utc_date": "2026-05-10T14:00:00Z",
        },
        {
            "match_id": "trois_heures",
            "utc_date": "2026-05-10T15:00:00Z",
        },
        {
            "match_id": "trente_minutes",
            "utc_date": "2026-05-10T17:30:00Z",
        },
        {
            "match_id": "futur",
            "utc_date": "2026-05-10T19:00:00Z",
        },
        {
            "match_id": "cible",
            "utc_date": "2026-05-09T12:00:00Z",
        },
        {
            "match_id": "inconnu",
            "utc_date": "",
        },
    ]

    expected = {
        "ancien",
        "cinq_heures",
    }

    print("\n=== TEST DES TROIS HISTORIQUES ===")

    for scope in (
        "HOME_HISTORY",
        "AWAY_HISTORY",
        "LEAGUE_HISTORY",
    ):
        original_rows = (
            prematch.filter_before_cutoff(
                FakeLab(),
                history,
                target,
                target_match_id="cible",
            )
        )

        guarded_rows = (
            adapter.__globals__[
                "filter_before_cutoff"
            ](
                FakeLab(),
                history,
                target,
                target_match_id="cible",
            )
        )

        original_ids = {
            r["match_id"]
            for r in original_rows
        }

        guarded_ids = {
            r["match_id"]
            for r in guarded_rows
        }

        assert guarded_ids == expected
        assert guarded_ids <= original_ids

        print(
            scope,
            "| original =", len(original_ids),
            "| V4.1 =", len(guarded_ids),
            "| exclus en plus =",
            len(original_ids - guarded_ids),
            "| OK",
        )

    print("\n=== VERIFICATION DE L'ISOLATION ===")

    assert (
        prematch.build_prematch_asof.__globals__[
            "filter_before_cutoff"
        ]
        is prematch.filter_before_cutoff
    )

    assert (
        adapter.__globals__[
            "filter_before_cutoff"
        ]
        is filter_before_cutoff_v41
    )

    print("Filtre original inchange : OK")
    print("Adaptateur V4.1 isole : OK")

    print("\n=== LIMITES ===")
    print(
        "Tests du filtre avec des rencontres "
        "synthetiques uniquement."
    )
    print(
        "Reconstruction PRE-MATCH complete "
        "non executee."
    )
    print(
        "Disponibilite reelle des statistiques "
        "non certifiee."
    )
    print(
        "Exclusions protegees non certifiees."
    )
    print("Aucun appel API.")
    print("Aucun dataset modifie.")
    print("TEST_FUTUR non consulte.")
    print("Scanner V3-C inchange.")
    print("ETAPE 313 TERMINEE.")


if __name__ == "__main__":
    print("=" * 78)
    print("ETAPE 313 - ADAPTATEUR PRE-MATCH V4.1 ISOLE")
    print("=" * 78)
    offline_test()
