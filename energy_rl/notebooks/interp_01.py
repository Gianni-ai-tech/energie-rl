INTERP = {
"uci": """**Lecture.** 2 075 259 mesures à la minute sur près de 4 ans, avec seulement 1,25 % de trous. Agrégées à l'heure, elles donnent 34 589 heures. Les trous sont bouchés par la même heure de la semaine précédente, puis par interpolation. La maison consomme en moyenne 1,09 kWh par heure, dont 0,39 kWh pour le chauffe-eau.""",

"ecs": """**Lecture.**
- **Un gros poste flexible :** le chauffe-eau consomme 9,3 kWh par jour en médiane, soit 36 % de la conso de la maison. C'est le gisement à piloter.
- **Il chauffe à la demande :** son pic est entre 7 h et 10 h, juste après les douches, et il ne profite pas des heures creuses (zones grises). C'est la référence « thermostat » du projet.
- **Résistance de 1,08 kW :** c'est la puissance la plus fréquente. Le simulateur suppose un ballon de 300 L à 3 kW, plus classique pour un pilotage en heures creuses (hypothèse documentée).
- **Creux en juillet-août :** probablement les vacances. Le reste de la maison double en hiver.""",

"pv": """**Lecture.**
- **Production annuelle :** 3 245 à 3 720 kWh selon l'année, soit 1 163 kWh par kWc en moyenne.
- **Saisonnalité forte :** environ 410 kWh par mois de mai à juillet, mais 120 kWh en décembre et janvier, justement les mois des jours rouges.
- **Conséquence pour le pilote :** en été, le levier est de chauffer l'eau à midi avec le solaire ; en hiver, il faut surtout déplacer la conso vers les heures creuses et utiliser la batterie.""",

"tempo": """**Lecture.**
- **Saisons régulières :** 300 jours bleus, 43 blancs et 22 rouges chaque saison, sauf 2019-2020 (47 blancs, 18 rouges).
- **Calendrier des jours rouges :** ils tombent de novembre à mars, surtout en janvier (127 sur 260), et jamais le week-end.
- **Écart de prix :** de 0,1356 € (heure creuse bleue) à 0,7295 € (heure pleine rouge), soit 5,4 fois plus cher. En dehors des jours rouges, l'écart entre heures pleines et creuses n'est que de 22 à 25 %.
- **À retenir :** l'essentiel du gain possible se joue sur 22 jours par an, de 6 h à 22 h.""",

"jeu": """**Lecture.** 78 882 heures sans valeur manquante, de septembre 2014 à août 2023. Selon la saison, la maison consomme 5 800 à 6 750 kWh hors chauffe-eau, et 3 040 à 3 600 kWh pour le chauffe-eau. Le solaire produit 3 240 à 3 820 kWh. La colonne `jour_source_uci` trace le jour réel rejoué. Les saisons 2014-2015 à 2020-2021 servent à l'apprentissage, 2021-2022 au choix du modèle, 2022-2023 au résultat final, jamais vue avant.""",
}
