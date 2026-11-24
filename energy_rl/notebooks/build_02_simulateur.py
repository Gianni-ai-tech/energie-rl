"""Construit 02_simulateur_regles.ipynb. Interprétations dans interp_02.py."""
import nbformat as nbf, importlib.util, os

INTERP = {}
if os.path.exists("interp_02.py"):
    spec = importlib.util.spec_from_file_location("i", "interp_02.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); INTERP = m.INTERP

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
lecture = lambda k: md(INTERP.get(k, f"**Lecture.** _à rédiger ({k})_"))

md("""
# Pilotage énergétique par RL – 2. Le simulateur et les règles de référence

Avant d'entraîner un agent, il faut deux choses : un **simulateur** crédible de la maison, et des **points de comparaison**. Un agent RL ne vaut que s'il fait mieux qu'une règle simple qu'un installateur pourrait écrire.

Le code est dans `energy_env.py` (classe `Maison` pour la physique, `MaisonEnv` pour l'interface Gymnasium, et les trois règles).
""")

code("""
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from dataclasses import asdict
from energy_env import Maison, Params, charger_donnees, simuler, bilan, REGLES, PRIX
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110

df = charger_donnees(); m = Maison(df)
pd.Series(asdict(Params()), name="valeur").to_frame()
""")
md("""
## 1. Ce que simule chaque heure

1. Le **besoin d'eau chaude** réel de l'heure (sous-compteur 3) est prélevé dans le ballon. Si le ballon ne suffit pas, l'énergie manquante est comptée comme « eau froide ».
2. Si le pilote a décidé de **chauffer**, la résistance de 3 kW remplit le ballon, sans dépasser sa capacité. Sous 1 kWh, une **sécurité de thermostat** force la chauffe, quel que soit le pilote : c'est le cas sur un vrai ballon, et ça met tous les pilotes à égalité.
3. Le **solaire** couvre d'abord la maison. Le surplus va dans la **batterie**, et ce qui reste part sur le réseau sans rémunération (hypothèse : autoconsommation sans revente).
4. Selon le **mode de batterie** choisi :
   - *autoconsommation* : elle se décharge pour couvrir le déficit ;
   - *charger* : elle se remplit aussi depuis le réseau ;
   - *conserver* : elle ne se décharge pas.
5. Le reste est **acheté** au prix Tempo de l'heure.
""")
code("""
# une heure de démonstration : un jour rouge de janvier à 8 h, ballon à moitié plein, batterie pleine
t = int(np.where((df.tempo == "rouge") & (df.index.hour == 8) & (df.index.month == 1))[0][0])
print(df.index[t], "| couleur :", df.tempo.iloc[t], "| prix :", m.prix[t], "€/kWh | besoin eau chaude :", round(float(m.ecs[t]), 2), "kWh")
for nom, (chauffe, mode) in {"ne rien chauffer, batterie en autoconsommation": (0, 0), "chauffer, batterie conservée": (1, 2)}.items():
    b, s, info = m.pas(t, 3.5, 5.0, chauffe, mode)
    print(f"- {nom} : achat {info['achat']:.2f} kWh, coût {info['cout']:.2f} €, ballon après {b:.2f} kWh, batterie après {s:.2f} kWh")
""")
lecture("demo")

md("## 2. Les trois règles de référence")
md("""
| Règle | Ballon | Batterie |
|---|---|---|
| **Thermostat** (la maison réelle) | se réchauffe dès que de l'eau est tirée | autoconsommation |
| **Contacteur heures creuses** (le standard) | chauffe de 22 h à 6 h (plus la sécurité sous 1 kWh) | autoconsommation |
| **Règle experte** (écrite avec le tarif en tête) | heures creuses + surplus solaire > 1,5 kW, évite les heures pleines rouges | chargée la nuit avant un jour blanc ou rouge, gardée pour les heures pleines |
""")
code("""
def saison(s):
    pos = np.where(df.saison == s)[0]; return int(pos[0]), int(pos[-1]) + 1

res, traces = {}, {}
for s in ["2021-2022", "2022-2023"]:
    a, b = saison(s)
    for nom, regle in REGLES.items():
        r = simuler(m, a, b, regle); traces[(s, nom)] = r
        res[(s, nom)] = bilan(r, m)
tab = pd.DataFrame(res).T
tab.index.names = ["saison", "règle"]
ref = tab.xs("thermostat", level="règle").facture_eur
tab["gain_vs_thermostat_%"] = (100 * (1 - tab.facture_eur / ref.reindex(tab.index.get_level_values(0)).values)).round(1)
tab
""")
lecture("resultats")

md("## 3. Quatre jours de décembre 2022, dont trois jours rouges d'affilée")
code("""
a, b = saison("2022-2023")
cal = pd.read_csv("../../data/tempo_couleurs.csv", parse_dates=["date"]).set_index("date").couleur
debut = pd.Timestamp("2022-12-11")
print("Couleurs :", {str(d.date()): cal[d] for d in pd.date_range(debut, periods=4)})
fen = slice(debut + pd.Timedelta(hours=6), debut + pd.Timedelta(days=4, hours=5))
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
d = df.loc[fen]
axes[0].step(d.index, d.prix, where="post", color="black"); axes[0].set(ylabel="€/kWh", title="Prix Tempo de l'heure")
for ax, nom in zip(axes[1:], ["thermostat", "heures_creuses", "experte"]):
    r = traces[("2022-2023", nom)].loc[fen]
    ax.fill_between(r.index, r.ballon, step="post", alpha=.3, label="énergie dans le ballon (kWh)")
    ax.bar(r.index, r.achat, width=1 / 24, align="edge", color="#c44e52", alpha=.7, label="achat réseau (kWh)")
    ax.plot(r.index, r.soc, color="#55a868", label="batterie (kWh)")
    ax.set(title=f"{nom} : {r.cout.sum():.2f} € sur ces 4 jours", ylabel="kWh"); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.show()
""")
lecture("semaine")

md("## 4. Où part l'argent ?")
code("""
lignes = []
for nom in REGLES:
    r = traces[("2022-2023", nom)]; d = df.loc[r.index]
    cle = d.tempo + np.where(d.hc, " HC", " HP")
    lignes.append(r.cout.groupby(cle).sum().rename(nom))
cout = pd.concat(lignes, axis=1).round(0)
cout.loc["total"] = cout.sum()
cout
""")
lecture("ventilation")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "02_simulateur_regles.ipynb")
print("ok", len(C), "cellules")
