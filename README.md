# Pilotage énergétique par apprentissage par renforcement (Jedha CDSD, bloc 6)

Un agent RL décide chaque heure s'il faut chauffer le ballon d'eau chaude et charger ou décharger la batterie d'une maison au tarif **Tempo**, pour réduire la facture sans priver la maison d'eau chaude.

| Lien | Adresse |
|---|---|
| Tableau de bord (démo en ligne) | https://gianni2504-pilotage-tempo-rl.static.hf.space |
| Code (ce dépôt) | https://github.com/Gianni-ai-tech/energie-rl |

## Résultat (saison test sept. 2022 – août 2023, jamais vue pendant l'apprentissage)

| Pilote | Facture | vs thermostat | Solaire autoconsommé | Heures d'eau tiède |
|---|---|---|---|---|
| Thermostat (la maison réelle) | 1 411 € | – | 87,0 % | 0 |
| Contacteur heures creuses | 1 329 € | −5,8 % | 81,6 % | 10 |
| Règle experte (écrite à la main) | 1 252 € | −11,3 % | 85,5 % | 8 |
| **Agent RL (PPO)** | **1 244 €** | **−11,9 %** | **91,1 %** | **4** |
| Optimum théorique (tout le futur connu) | 1 131 € | −19,9 % | – | 0 |

L'agent bat la règle experte sur 8 saisons sur 9 (+9,4 € par saison en moyenne). Le gain au-delà de l'expert reste faible (0,7 % sur le test), et il reste environ 10 % d'écart avec l'optimum théorique.

## Données (publiques)

| Source | Usage | Licence |
|---|---|---|
| [UCI – Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption) (EDF R&D) | conso à la minute d'une maison de Sceaux, 2006-2010 ; le sous-compteur 3 donne le besoin d'eau chaude | CC BY 4.0 |
| [PVGIS](https://re.jrc.ec.europa.eu/pvg_tools/fr/) (Commission européenne) | production horaire d'un 3 kWc à Sceaux, 2014-2023 | libre avec citation |
| [Jour Tempo](https://www.data.gouv.fr/organizations/jour-tempo) (couleurs RTE) | couleur de chaque jour depuis 2014 | Licence Ouverte 2.0 |
| Grille Tempo TTC au 1er août 2026 (délibération CRE du 15/07/2026) | 6 prix du kWh | publique |

Le fichier brut UCI (127 Mo) n'est pas dans le dépôt : le télécharger depuis le lien ci-dessus et le placer dans `data/household_power_consumption.txt` pour relancer `prepare_data.py` et le notebook 1. Le jeu horaire prêt à l'emploi (`data/dataset_horaire.parquet`) est fourni.

## Contenu

```
data/                          jeu horaire, solaire PVGIS, couleurs Tempo
energy_rl/
  prepare_data.py              construit le jeu horaire 2014-2023
  energy_env.py                simulateur (ballon, batterie, solaire, prix), environnement Gymnasium, 3 règles
  oracle.py                    optimum théorique (programme linéaire, tout le futur connu)
  train.py                     PPO à partir de zéro (v1)
  train_v3.py                  imitation de la règle experte puis PPO (version finale : ppo_v4)
  continuer.py                 prolonge un entraînement
  export_dashboard.py          précalcule les simulations de la saison test pour le tableau de bord
  modeles/                     agent final (ppo_v4_meilleur.zip) et historiques d'apprentissage
  notebooks/                   01 données, 02 simulateur et règles, 03 agent (.ipynb exécutés + .html)
  dashboard/                   Streamlit (stlite), déployé sur un Space Hugging Face statique
```

## Relancer

```bash
pip install -r requirements.txt
cd energy_rl
python train_v3.py 2000000 ppo_v4 3e-4 0.01 0.2 10   # ~15 min sur 2 CPU
python export_dashboard.py ppo_v4
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 03_agent_rl.ipynb
cd ../dashboard && streamlit run app.py
```

## Hypothèses principales

- Ballon de 300 L à 3 kW (7 kWh utiles), pertes de 0,5 % par heure, sécurité du thermostat sous 1 kWh, identique pour tous les pilotes.
- Batterie de 5 kWh / 2,5 kW, rendement de 95 % par sens. Le surplus solaire n'est pas rémunéré.
- La couleur du lendemain est connue à partir de 11 h. Les prévisions solaires à 6 h et à 24 h sont supposées parfaites.
- La conso de 2006-2010 est rejouée sur le calendrier 2014-2023 : même semaine et même jour de la semaine.
- Découpage : apprentissage sur les saisons 2014-2015 à 2020-2021, choix du modèle sur 2021-2022, test sur 2022-2023.

Projet Jedha Data Science & Engineering Fullstack, bloc 6 (direction de projet data) – Gianni Murru.
