INTERP = {
"demo": """**Lecture.** Le 5 janvier 2015 à 8 h, jour rouge (0,7295 €/kWh), il faut 1,05 kWh d'eau chaude. Si le pilote ne chauffe pas et laisse la batterie couvrir la maison, il n'achète rien. S'il chauffe en gardant la batterie, il achète 4,79 kWh, soit 3,49 € pour une seule heure. Un mauvais choix à la mauvaise heure coûte cher : c'est ce que l'agent doit apprendre à éviter.""",

"resultats": """**Lecture (saison test 2022-2023).**

| Règle | Facture | Gain vs thermostat |
|---|---|---|
| Thermostat (maison réelle) | 1 411 € | référence |
| Contacteur heures creuses | 1 329 € | −5,8 % |
| Règle experte | 1 252 € | −11,3 % |

- **Où se fait le gain :** l'achat en heures pleines rouges passe de 438 à 250 kWh avec la règle experte.
- **Ce que coûte le contacteur :** l'autoconsommation solaire baisse (81,6 % contre 87 %), car le ballon est déjà plein quand le soleil arrive.
- **Confort :** les pannes d'eau chaude restent marginales, 8 à 25 heures par an pour moins de 2 kWh manquants.
- **Ce que l'agent doit battre :** la règle experte, 1 128 € sur la validation et 1 252 € sur le test.""",

"semaine": """**Lecture.** Sur ces 4 jours (un bleu puis trois rouges), la facture est de 54,55 € avec le thermostat, 45,90 € avec le contacteur et 38,76 € avec la règle experte.
- **Ce que fait l'expert :** il remplit la batterie la nuit (courbe verte à 5 kWh à 6 h) et chauffe l'eau en heures creuses.
- **Ce qui reste :** 35,1 kWh achetés en heures pleines rouges, surtout le soir (26,1 kWh entre 18 h et 22 h), une fois la batterie vidée. Garder la batterie pour le soir n'y changerait rien : de 6 h à 22 h, le prix est le même. Le seul levier serait une batterie plus grande. La marge laissée par l'expert se mesure donc avec l'optimum théorique (notebook 3).""",

"ventilation": """**Lecture.** Le poste qui change le plus est l'heure pleine rouge : 320 € avec le thermostat, 249 € avec le contacteur, 182 € avec la règle experte, sur 22 jours seulement. Les heures pleines bleues pèsent aussi (651 € avec le thermostat contre 435 € avec l'expert), car on y déplace la chauffe vers la nuit. En contrepartie, les heures creuses montent (bleu HC de 215 à 395 €) : on achète plus la nuit, mais moins cher.""",
}
