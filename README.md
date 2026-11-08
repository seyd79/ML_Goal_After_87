# Étape 10 — Fusion expérimentale V3-C + Pressure Features V2

Cette étape prépare **des tableaux comparables à la 87e minute**. Elle **n'entraîne pas** de modèle et n'active aucune alerte.

## Le principe

- Table **A** : 22 colonnes candidates de V3-C historique, à 87'. Ce n'est **pas** une reproduction complète des 96 features / du modèle V3-C déployé. Certaines statistiques historiques ont été reconstruites après les matchs et doivent être vérifiées.
- Table **B** : exactement les mêmes matchs et 22 colonnes **+ 6 nouvelles features Pressure V2**.
- **Labels** `goal_after_87` dans un fichier **séparé** : aucune colonne de cible ou de fin de match dans A/B.
- Les 585 snapshots Pressure V2 aux minutes 85/86/87 sont conservés dans un autre fichier. La baseline fournie n'existe qu'à 87'. Il est interdit de la joindre aux snapshots de 85'/86', ce qui introduirait des informations futures.

## Sources

- `historical_1710_baseline_87_REVIEW.csv` : données candidates extraites de l'historique V3-C déjà fourni, sans cibles ni scores finaux. Ne pas les considérer automatiquement fiables **en temps réel**.
- `historical_1710_manifest.csv` et `historical_1710_reference_labels_SEPARATE.csv` de l'étape 6 (sur le Mac de l'utilisateur).
- `outputs/pressure_historical/pressure_v2_experimental/pressure_v2_85_86_87_EXPERIMENTAL.csv` de l'étape 9 (sur le Mac de l'utilisateur).

## Fonctions de sécurité

- Jointure stricte (`match_id`, `minute`) et contrôle des doublons.
- La version historique de V3-C à 87' ne peut servir qu'à la paire A/B à 87'.
- Aucune utilisation des champs `final_score_*`, `goal_minutes_after_87`, `goal_after_87`, `target_*`, `model_probability` ou des bilans `/stats` en entrée du prédicteur.
- Vérification des ID, équipes, dates, valeurs manquantes, pressions négatives, tirs cadrés > tirs.
- Tous les labels de référence sont marqués `label_revalidated=0`. L'ancien travail avait relevé des incohérences sur certains labels historiques; nous ne les certifions pas ici.
- `training_ready=0` pour tous les matches, car la provenance temporelle historique reste à auditer.
- Zéro requête API, zéro écrasement des fichiers historiques.

## Commandes

Depuis la racine de `ML_Goal_After_87:` :

```bash
unzip -n ~/Downloads/V3C_Etape10_Fusion_Pressure_V2_2026-10-25.zip
./.venv/bin/python test_merge_v3c_pressure_v2.py
./.venv/bin/python merge_v3c_pressure_v2.py \
  --pressure outputs/pressure_historical/pressure_v2_experimental/pressure_v2_85_86_87_EXPERIMENTAL.csv \
  --manifest historical_1710_manifest.csv \
  --labels historical_1710_reference_labels_SEPARATE.csv \
  --baseline historical_1710_baseline_87_REVIEW.csv \
  --outdir outputs/pressure_historical/pressure_v2_dataset_join
cat outputs/pressure_historical/pressure_v2_dataset_join/MERGE_SUMMARY.json
```

**Résultats attendus si 195 matchs en cache et présence des 195 identifiants dans la baseline :** 585 lignes Pressure-only (85/86/87), 195 lignes de modèle A, 195 lignes de modèle B et 195 labels séparés. Ce sont des attentes, **pas des résultats obtenus sur le Mac**.

## Prochaine étape

1. Revalider les étiquettes (une cible par match, but strictement >87'), les incohérences de score et l'historique de mise à jour.
2. Vérifier le caractère **as-of** des variables cumulées de V3-C : les chiffres historiques reconstruits à 87' ne prouvent pas la disponibilité en direct. Revoir la minute exacte (87 affichée vs 87 révolue).
3. Conserver les mêmes matchs dans A et B, séparer train/validation/test **chronologiquement** avec des groupes par match, sans utiliser les 95 premiers matchs inspectés pour annoncer une évaluation indépendante des conventions V2.
4. Comparer la discrimination, la calibration, et le taux de réussite **avec couverture** sur un échantillon beaucoup plus important. Les 195 matchs restent peu nombreux pour valider 60 % de succès.
