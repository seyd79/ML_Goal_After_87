import inspect
from pathlib import Path

import rebuild_clean_prematch_v2 as prematch

from v41_prematch_temporal_adapter_313 import (
    build_isolated_adapter,
)


ODDS_FIELDS = (
    "odds_home",
    "odds_draw",
    "odds_away",
    "odds_over05",
    "odds_over15",
    "odds_over25",
    "odds_over35",
    "odds_under25",
    "odds_btts_yes",
    "odds_btts_no",
)


class SafeLabProxy:
    """
    Module LAB delegue, sauf pour les cotes.
    Aucune lecture du endpoint de cotes.
    """

    def __init__(self, original_lab):
        self._original = original_lab
        self.odds_calls_blocked = 0

    def fetch_prematch_odds(
        self,
        client,
        match_id,
        preferred_bookmaker=None,
    ):
        self.odds_calls_blocked += 1

        return {
            key: None
            for key in ODDS_FIELDS
        }

    def __getattr__(self, name):
        return getattr(self._original, name)


class ResearchPrematchProxy:
    """
    Proxy PRE-MATCH pour la recherche V4.1.

    - Filtre temporel 4 h
    - Cache odds independant
    - Odds neutralisees
    - xG supplementaire interdit
    - Seconde periode enrichie interdite
    """

    def __init__(self, original_module, original_lab):
        self._original = original_module
        self.lab = SafeLabProxy(original_lab)

        self._adapter = build_isolated_adapter()

        # Rebinding dans les globals de la copie
        # fonctionnelle uniquement.
        self._adapter.__globals__["ODDS_CACHE"] = {}

        original_cache = (
            original_module.build_prematch_asof
            .__globals__["ODDS_CACHE"]
        )

        isolated_cache = (
            self._adapter.__globals__["ODDS_CACHE"]
        )

        if isolated_cache is original_cache:
            raise RuntimeError(
                "STOP : cache de cotes encore partage"
            )

        if not isinstance(isolated_cache, dict):
            raise RuntimeError(
                "STOP : type de cache inattendu"
            )

        actual_odds_fields = {
            str(name)
            for name in original_module.PRE_MATCH_FEATURES
            if str(name).startswith("odds_")
        }

        if actual_odds_fields != set(ODDS_FIELDS):
            raise RuntimeError(
                "STOP : schema des cotes modifie"
            )

    def build_prematch_asof(
        self,
        lab,
        client,
        target_match,
        cutoff,
        xg_limit,
        with_second_half,
        preferred_bookmaker,
    ):
        # Bloquer les options non autorisees.
        if xg_limit != 0:
            raise ValueError(
                "V4.1 : xg_limit doit etre 0"
            )

        if with_second_half is not False:
            raise ValueError(
                "V4.1 : with_second_half doit etre False"
            )

        if lab is not self.lab:
            raise ValueError(
                "V4.1 : LAB non protege"
            )

        return self._adapter(
            lab=self.lab,
            client=client,
            target_match=target_match,
            cutoff=cutoff,
            xg_limit=0,
            with_second_half=False,
            preferred_bookmaker=preferred_bookmaker,
        )

    def __getattr__(self, name):
        return getattr(self._original, name)


def offline_test():
    print("=" * 78)
    print("ETAPE 318 - ISOLATION COTES ET OPTIONS V4.1")
    print("=" * 78)

    original_function = prematch.build_prematch_asof
    original_cache = (
        original_function.__globals__["ODDS_CACHE"]
    )

    class FakeLab:
        marker = "delegation_ok"

        def fetch_prematch_odds(self, *args, **kwargs):
            raise AssertionError(
                "Le LAB original ne doit pas etre appele"
            )

    lab = FakeLab()
    proxy = ResearchPrematchProxy(prematch, lab)

    isolated = proxy._adapter
    isolated_cache = isolated.__globals__["ODDS_CACHE"]

    print("\n=== ISOLATION ===")

    assert isolated is not original_function
    assert isolated_cache is not original_cache
    assert isolated_cache == {}

    assert (
        original_function.__globals__["ODDS_CACHE"]
        is original_cache
    )

    assert (
        inspect.signature(isolated)
        == inspect.signature(original_function)
    )

    assert (
        isolated.__globals__["filter_before_cutoff"]
        is not prematch.filter_before_cutoff
    )

    assert proxy.lab.marker == "delegation_ok"

    print("Constructeur original intact : OK")
    print("Cache original conserve : OK")
    print("Cache V4.1 distinct et vide : OK")
    print("Filtre temporel V4.1 conserve : OK")
    print("Delegation LAB : OK")

    print("\n=== NEUTRALISATION DES COTES ===")

    class ForbiddenClient:
        def get(self, *args, **kwargs):
            raise AssertionError(
                "Aucun appel API autorise"
            )

    client = ForbiddenClient()

    odds = proxy.lab.fetch_prematch_odds(
        client,
        "match_synthetique",
        preferred_bookmaker=None,
    )

    assert set(odds) == set(ODDS_FIELDS)
    assert all(value is None for value in odds.values())
    assert proxy.lab.odds_calls_blocked == 1

    # Le cache V4.1 peut contenir uniquement
    # ses propres entrees ; aucune ecriture
    # dans le cache original.
    assert isolated_cache == {}
    assert (
        original_function.__globals__["ODDS_CACHE"]
        is original_cache
    )

    print("Nombre de champs odds :", len(odds))
    print("Valeurs toutes manquantes : OK")
    print("Aucun appel au LAB original : OK")
    print("Aucun appel API : OK")

    print("\n=== OPTIONS INTERDITES ===")

    kwargs = {
        "lab": proxy.lab,
        "client": client,
        "target_match": {},
        "cutoff": "2026-05-10T18:00:00Z",
        "xg_limit": 0,
        "with_second_half": False,
        "preferred_bookmaker": None,
    }

    for changed, value in (
        ("xg_limit", 1),
        ("with_second_half", True),
        ("lab", lab),
    ):
        test_kwargs = dict(kwargs)
        test_kwargs[changed] = value

        try:
            proxy.build_prematch_asof(**test_kwargs)
        except ValueError:
            print(changed, ": correctement bloque")
        else:
            raise AssertionError(
                "Option non bloquee : " + changed
            )

    print("\n=== CONTROLE DES FICHIERS ===")

    assert Path(
        "v41_prematch_temporal_adapter_313.py"
    ).is_file()

    print("Adaptateur et module original conserves : OK")

    print("\n=== LIMITES ===")
    print("Branchement et neutralisation testes hors ligne.")
    print("Aucune reconstruction complete effectuee.")
    print("Disponibilite des autres donnees non certifiee.")
    print("Exclusions protegees non certifiees.")
    print("Collecte des 12 matchs : BLOQUEE")
    print("Aucun appel API.")
    print("Aucun dataset modifie.")
    print("Aucun holdout ouvert.")
    print("TEST_FUTUR non consulte.")
    print("V3-C inchange.")
    print("ETAPE 318 TERMINEE.")


if __name__ == "__main__":
    offline_test()
