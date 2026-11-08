G02-G38 | TRIAGE DE PROVENANCE DES ARCHIVES V3C (LECTURE SEULE)

1) Depuis ML_Goal_After_87, extraire le ZIP (ou deposer le .py).
2) python3 -B g02g38_archive_provenance_static_readonly.py
3) Si G02-G38/P1 PRECHECK PASS :
   python3 -B g02g38_archive_provenance_static_readonly.py --audit-provenance

DEPENDANCES : les scripts inchanges G02-G33 a G02-G37 et les artefacts
V4.2 geles doivent etre presents dans le dossier du projet.

CONTROLES : intersections d'identifiants seulement sur les 1500 matchs
V4.2, anomalies minute/UTC des anciennes archives, recherche statique
dans les scripts Python (aucune execution). Labels, predictions, scores
et timelines fermes. Aucun appel API, ecriture ni fit.

ATTENTION : une capture datee n'est pas automatiquement certifiee as-of.
Les lignes de code ne sont PAS imprimees, seulement noms de fichiers et
positions des occurrences. Les indices de fichier ne prouvent pas l'origine
de l'horodatage.
