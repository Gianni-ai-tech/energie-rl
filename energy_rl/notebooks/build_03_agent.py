"""Construit 03_agent_rl.ipynb. Interprétations dans interp_03.py.
L'entraînement (train.py, train_v3.py) n'est pas relancé ici : le notebook charge les modèles et historiques sauvegardés."""
import nbformat as nbf, importlib.util, os, sys

MODELE = sys.argv[1] if len(sys.argv) > 1 else "ppo_v3b"
INTERP = {}
if os.path.exists("interp_03.py"):
    spec = importlib.util.spec_from_file_location("i", "interp_03.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); INTERP = m.INTERP

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
lecture = lambda k: md(INTERP.get(k, f"**Lecture.** _à rédiger ({k})_"))

md("""
# Pilotage énergétique par RL – 3. L'agent

**Le problème posé à l'agent.** Chaque heure, il voit l'heure, la couleur du jour et celle du lendemain (dès 11 h), le niveau du ballon et de la batterie, le solaire des 6 prochaines heures et les prix à venir. Il choisit 1 action sur 6 : chauffer ou non le ballon, et mettre la batterie en autoconsommation, en charge ou en conservation. Sa récompense est la facture de l'heure, en négatif, moins 3 € par kWh d'eau chaude manquante.

**L'algorithme.** PPO (Stable-Baselines3), un réseau de 2 × 128 neurones. Deux façons de l'entraîner ont été comparées :
1. **v1, à partir de zéro** (`train.py`) : 1,5 million d'heures simulées, avec une première version des indicateurs (sans les prix à venir ni la prévision solaire du lendemain).
2. **Version finale, imitation puis RL** (`train_v3.py`, modèle `ppo_v4`) : le réseau apprend d'abord à reproduire la règle experte sur les 7 saisons d'entraînement, puis PPO l'améliore pendant 2 millions d'heures. L'agent voit aussi les prix à venir et la prévision solaire du lendemain. Le meilleur modèle est choisi sur 2021-2022.

Une sécurité de thermostat force la chauffe sous 1 kWh, pour tous les pilotes.

La saison 2022-2023 n'a servi à rien pendant l'entraînement : c'est le test.
""")

code(f"""
import sys, json; sys.path.insert(0, "..")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from stable_baselines3 import PPO
from energy_env import Maison, charger_donnees, simuler, bilan, REGLES
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110
MOD = "../modeles"
df = charger_donnees(); m = Maison(df)
agent = PPO.load(f"{{MOD}}/{MODELE}_meilleur", device="cpu")
def saison(s):
    pos = np.where(df.saison == s)[0]; return int(pos[0]), int(pos[-1]) + 1
""")

md("## 1. L'apprentissage")
code(f"""
h1 = pd.DataFrame(json.load(open(f"{{MOD}}/ppo_v1_historique.json")))
h3 = pd.DataFrame(json.load(open(f"{{MOD}}/{MODELE}_historique.json")))
a, b = saison("2021-2022")
ref = {{k: bilan(simuler(m, a, b, f), m)["facture_eur"] for k, f in REGLES.items()}}
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(h1.pas / 1e6, h1.facture_eur, marker=".", label="v1 : PPO à partir de zéro")
ax.plot(h3.pas / 1e6, h3.facture_eur, marker=".", label="version finale : imitation de l'expert puis PPO")
for k, c in zip(ref, ["grey", "#8172b2", "black"]):
    ax.axhline(ref[k], ls="--", color=c, lw=1, label=f"règle {{k}} ({{ref[k]:.0f}} €)")
ax.set(title="Facture sur la saison de validation 2021-2022 pendant l'apprentissage", xlabel="millions d'heures simulées", ylabel="€", ylim=(1050, 1450))
ax.legend(fontsize=8, loc="upper right"); plt.tight_layout(); plt.show()
print("v1 : meilleure facture", h1.facture_eur.min(), "€ | eau froide à ce point :", int(h1.loc[h1.score.idxmin(), "eau_froide_h"]), "h")
print("version finale : facture après imitation", h3.facture_eur.iloc[0], "€ | meilleur score", h3.score.min(), "à", int(h3.loc[h3.score.idxmin(), "pas"]), "pas")
""")
lecture("apprentissage")

md("## 2. Le résultat sur la saison test 2022-2023")
code("""
res, traces = {}, {}
for s in ["2021-2022", "2022-2023"]:
    a, b = saison(s)
    for nom, f in {**REGLES, "agent RL": agent}.items():
        r = simuler(m, a, b, f); traces[(s, nom)] = r; res[(s, nom)] = bilan(r, m)
tab = pd.DataFrame(res).T; tab.index.names = ["saison", "pilote"]
for s in ["2021-2022", "2022-2023"]:
    th, ex = tab.loc[(s, "thermostat"), "facture_eur"], tab.loc[(s, "experte"), "facture_eur"]
    tab.loc[s, "gain_vs_thermostat_%"] = (100 * (1 - tab.loc[s, "facture_eur"] / th)).round(1).values
    tab.loc[s, "gain_vs_experte_eur"] = (ex - tab.loc[s, "facture_eur"]).round(1).values
tab
""")
lecture("resultat")

md("## 3. Ce que l'agent fait différemment")
code("""
fin = slice(pd.Timestamp("2022-12-11 06:00"), pd.Timestamp("2022-12-15 05:00"))
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
d = df.loc[fin]
axes[0].step(d.index, d.prix, where="post", color="black"); axes[0].set(ylabel="€/kWh", title="Prix Tempo (11 déc. bleu, puis 3 jours rouges)")
for ax, nom in zip(axes[1:], ["experte", "agent RL"]):
    r = traces[("2022-2023", nom)].loc[fin]
    ax.fill_between(r.index, r.ballon, step="post", alpha=.3, label="ballon (kWh)")
    ax.bar(r.index, r.achat, width=1 / 24, align="edge", color="#c44e52", alpha=.7, label="achat réseau (kWh)")
    ax.plot(r.index, r.soc, color="#55a868", label="batterie (kWh)")
    ax.set(title=f"{nom} : {r.cout.sum():.2f} € sur ces 4 jours", ylabel="kWh"); ax.legend(loc="upper right", fontsize=8)
plt.tight_layout(); plt.show()
for nom in ["experte", "agent RL"]:
    r = traces[("2022-2023", nom)].loc[fin]; dd = df.loc[r.index]
    rhp = (dd.tempo == "rouge") & ~dd.hc
    print(f"{nom} : {r.achat[rhp].sum():.1f} kWh achetés en heures pleines rouges, dont {r.achat[rhp & (dd.index.hour >= 18)].sum():.1f} kWh entre 18 h et 22 h")
""")
lecture("semaine")

code("""
r = traces[("2022-2023", "agent RL")]; d = df.loc[r.index]
x = pd.DataFrame({"heure": d.index.hour, "couleur": d.tempo.values, "chauffe": ((r.action // 3 == 1) | (r.ballon < 1.0)).values,
                  "mode": (r.action % 3).map({0: "autoconsommation", 1: "charger", 2: "conserver"}).values})
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
x.pivot_table(index="couleur", columns="heure", values="chauffe", aggfunc="mean").loc[["bleu", "blanc", "rouge"]].pipe(
    sns.heatmap, ax=axes[0], cmap="Blues", vmin=0, vmax=1, cbar_kws={"label": "part des heures où le ballon chauffe"})
axes[0].set(title="Agent : quand chauffe-t-il l'eau ?")
pd.crosstab([x.couleur], x["mode"], normalize="index").loc[["bleu", "blanc", "rouge"]].mul(100).round(1).plot.barh(stacked=True, ax=axes[1])
axes[1].set(title="Agent : mode de la batterie selon la couleur du jour", xlabel="% des heures"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
lecture("comportement")

md("## 4. Est-ce que ça tient sur toutes les saisons ?")
code("""
lignes = []
for s in sorted(df.saison.unique()):
    a, b = saison(s)
    ex = bilan(simuler(m, a, b, REGLES["experte"]), m); ag = bilan(simuler(m, a, b, agent), m)
    th = bilan(simuler(m, a, b, REGLES["thermostat"]), m)
    lignes.append({"saison": s, "rôle": "entraînement" if s < "2021-2022" else ("validation" if s == "2021-2022" else "test"),
                   "thermostat_eur": th["facture_eur"], "experte_eur": ex["facture_eur"], "agent_eur": ag["facture_eur"],
                   "agent_vs_experte_eur": round(ex["facture_eur"] - ag["facture_eur"], 1),
                   "eau_froide_h_agent": ag["eau_froide_h"]})
gen = pd.DataFrame(lignes).set_index("saison")
gen
""")
lecture("generalisation")

md("""## 5. Et si la prévision solaire est fausse ?

L'agent a été entraîné avec le solaire des 6 et 24 prochaines heures connu exactement. On le réévalue sans le réentraîner (`robustesse_prevision.py`) :
- **prévision bruitée** : la production de chaque jour prévu est multipliée par un facteur aléatoire (erreur de 10, 25 ou 50 %), 5 tirages par niveau ;
- **persistance** : « demain comme aujourd'hui », c'est-à-dire aucune prévision météo ;
- **contrôle** : un agent entraîné exactement pareil, mais sans la prévision à 24 h (`ablation_sans24h.py`).

La règle experte n'utilise aucune prévision : son résultat ne change pas.""")
code("""
rob = pd.read_csv("../resultats_robustesse.csv")
(rob.groupby(["saison", "scenario"], sort=False)["facture_eur"].agg(["mean", "min", "max"]).round(1)
    .rename(columns={"mean": "facture moyenne (€)", "min": "min", "max": "max"}))
""")
lecture("robustesse")

md("## 6. Limites et suite")
lecture("limites")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "03_agent_rl.ipynb")
print("ok", len(C), "cellules")
