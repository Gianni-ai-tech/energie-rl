---
title: Pilotage Tempo par RL
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: static
pinned: false
---

# Pilotage d'un chauffe-eau et d'une batterie au tarif Tempo par apprentissage par renforcement

Compare, sur la saison 2022-2023, un agent RL (PPO) à trois règles de pilotage (thermostat, contacteur heures creuses, règle experte) : facture annuelle, achats en heures pleines rouges, autoconsommation solaire, confort. On peut rejouer n'importe quelle journée heure par heure.

Application Streamlit exécutée directement dans le navigateur avec [stlite](https://github.com/whitphx/stlite) (premier chargement : 20 à 40 secondes).

Projet Jedha (bloc 6, CDSD) – Gianni Murru. Code : https://github.com/Gianni-ai-tech/energie-rl
