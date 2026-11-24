INTERP = {
"apprentissage": """**Lecture.**
- **v1, à partir de zéro :** l'agent part de très haut (1 344 € à 50 000 heures). Il passe sous le thermostat (1 302 €), puis sous le contacteur heures creuses (1 224 €), mais plafonne vers 1 133 € après 1,5 million d'heures, avec 27 heures d'eau froide. Il n'atteint jamais la règle experte.
- **Version finale, imitation puis RL :** l'agent reproduit d'abord l'expert (98,4 % de décisions identiques, 1 129 €). Ensuite PPO l'améliore jusqu'à 1 105 € (score de 1 110 € avec la pénalité d'eau froide), à 1,95 million d'heures.
- **Ce qui a débloqué la situation :** donner à l'agent la prévision solaire du lendemain. Sans elle, imitation puis PPO restait collé à l'expert (essais v3, non montrés).
- **Pourquoi la courbe est bruitée :** le score est celui d'une saison entière, et une petite modification de la politique déplace des centaines de décisions. On garde le meilleur modèle sur la validation.""",

"resultat": """**Lecture (saison test 2022-2023, jamais vue).**

| Pilote | Facture | vs thermostat | vs contacteur HC | vs règle experte |
|---|---|---|---|---|
| Thermostat (maison réelle) | 1 411 € | – | | |
| Contacteur heures creuses | 1 329 € | −5,8 % | – | |
| Règle experte | 1 252 € | −11,3 % | −5,8 % | – |
| **Agent RL** | **1 244 €** | **−11,9 %** | **−6,4 %** | **−8,5 € (−0,7 %)** |
| Optimum théorique (tout le futur connu) | 1 131 € | −19,9 % | | |

- **Gain réel :** 168 € par an face au thermostat. L'agent fait à peine mieux que la règle experte : il ne rattrape que 7 % de l'écart entre l'expert et l'optimum théorique.
- **Comment il gagne :** il achète 204 kWh de moins (7 014 contre 7 218) et autoconsomme 91,1 % du solaire contre 85,5 %. Les heures pleines rouges restent au même niveau (249 kWh).
- **Confort :** 4 heures d'eau tiède dans l'année, pour 0,11 kWh manquant.
- **Écart avec l'objectif de départ :** la cible était −10 % face au contacteur heures creuses, on obtient −6,4 %. Elle n'est pas atteinte.""",

"semaine": """**Lecture.** Sur ces 4 jours, l'agent (38,12 €) et l'expert (38,76 €) font presque pareil, mais pas de la même façon.
- **Le défaut visible :** l'agent fait de petites relances du ballon tôt le matin des jours rouges. Sur la saison, entre 6 h et 8 h, il chauffe 30,5 kWh, dont 21 kWh achetés au réseau, soit 15,3 € contre 1,5 € pour l'expert.
- **Pourquoi on ne le corrige pas à la main :** lui interdire après coup de chauffer en heure pleine rouge dégrade son résultat de 5 € sur la saison (1 248,6 € au lieu de 1 243,7 €). Ses décisions forment un tout. La bonne correction serait de réentraîner avec cette contrainte.""",

"comportement": """**Lecture.**
- **Chauffe :** l'agent chauffe surtout la nuit (heures creuses) les jours blancs et rouges, et vers midi les jours bleus, quand le solaire produit (13 h - 16 h). Les heures 0 h - 6 h comptées « rouges » sont la nuit qui suit un jour rouge, au tarif heures creuses.
- **Batterie :** les jours bleus, l'agent la laisse en *conserver* 40 % du temps, pour garder le solaire pour plus tard au lieu de la vider tout de suite. Les jours rouges, il la *charge* 16 % du temps, la nuit avant les heures pleines.
- **Ce qui le distingue de l'expert :** il chauffe beaucoup moins la nuit (2,3 à 4,4 kWh par nuit en moyenne selon le soleil du lendemain, contre 4,5 à 6,1 pour l'expert). Il garde de la place dans le ballon pour le solaire de la journée, d'où 91 % d'autoconsommation contre 85,5 %. Les deux pilotes chauffent moins la nuit quand le lendemain est ensoleillé (corrélation de −0,31 et −0,33).""",

"generalisation": """**Lecture.**
- **Régularité :** l'agent fait mieux que la règle experte sur 8 saisons sur 9, de 9,4 € par saison en moyenne (de −2,3 € à +24,8 €).
- **Pas de surapprentissage visible :** le gain n'est pas plus grand sur les saisons d'entraînement que sur le test.
- **Effet de sélection :** la validation (+22,3 €) a servi à choisir le modèle, son chiffre est donc flatteur. Le test (+8,5 €) est le chiffre honnête.
- **Confort :** les heures d'eau tiède (4 à 35 par an) restent du même ordre que celles des règles.""",

"limites": """**Ce que montre le projet.** Un agent RL apprend seul une stratégie au moins aussi bonne qu'une règle écrite par un expert, sans qu'on lui code le tarif. C'est l'intérêt pour un fournisseur : si la grille Tempo ou l'équipement change, on réentraîne, on ne réécrit pas de règle. Mais sur ce cas, le gain au-delà de l'expert est faible (0,7 %), et une règle bien pensée capte déjà l'essentiel.

**Limites assumées.**
- **Une seule maison, une seule saison test :** le gain de 8,5 € est trop petit pour être affirmé avec certitude. Il faudrait tester sur d'autres foyers.
- **Prévisions supposées parfaites :** le solaire à 24 h est connu exactement. Avec une vraie prévision météo, le gain serait plus faible.
- **Simulateur simplifié :** pertes du ballon proportionnelles, rendement de batterie fixe, pas de vieillissement.
- **Décalage de dates :** la conso de 2006-2010 est rejouée sur le calendrier 2014-2023.

**Pistes.**
1. Réentraîner avec l'interdiction de chauffer en heure pleine rouge (contrainte apprise, pas plaquée).
2. Une prévision solaire bruitée, pour mesurer la perte réelle.
3. Plusieurs maisons (jeux de données de type Linky, avec consentement).
4. Tester sur un vrai ballon piloté par un boîtier domotique.""",
}
