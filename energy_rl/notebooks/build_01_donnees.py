"""Construit 01_donnees.ipynb. Interprétations dans interp_01.py."""
import nbformat as nbf, importlib.util, os

INTERP = {}
if os.path.exists("interp_01.py"):
    spec = importlib.util.spec_from_file_location("i", "interp_01.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); INTERP = m.INTERP

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
lecture = lambda k: md(INTERP.get(k, f"**Lecture.** _à rédiger ({k})_"))

md("""
# Pilotage énergétique par RL – 1. Les données

**Le besoin.** Au tarif Tempo, l'heure pleine d'un jour rouge coûte 0,7295 €/kWh, 5,4 fois l'heure creuse d'un jour bleu. Un ballon d'eau chaude et une batterie peuvent décaler de la consommation, à condition de savoir **quand**. On veut un agent qui le décide seul, heure par heure.

**Ce notebook** rassemble les trois sources publiques, vérifie leur qualité et construit le jeu horaire du simulateur (`prepare_data.py`) :

| Source | Contenu | Licence |
|---|---|---|
| UCI (EDF R&D) | conso à la minute d'une maison de Sceaux, déc. 2006 – nov. 2010 | CC BY 4.0 |
| PVGIS (Commission européenne) | production horaire simulée d'un 3 kWc à Sceaux, 2014 – 2023, et température | libre avec citation |
| Jour Tempo (données RTE) | couleur de chaque jour depuis le 01/09/2014 | Licence Ouverte 2.0 |
""")

code("""
import sys; sys.path.insert(0, "..")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from prepare_data import uci_hourly, pvgis_local, DATA
from energy_env import PRIX
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 110

brut = pd.read_csv(DATA / "household_power_consumption.txt", sep=";", na_values="?", low_memory=False)
print(f"UCI : {len(brut):,} mesures à la minute, du {brut.Date.iloc[0]} au {brut.Date.iloc[-1]}")
print(f"Valeurs manquantes : {brut.Global_active_power.isna().mean():.2%}")
h = uci_hourly()
print(f"Agrégé à l'heure : {len(h):,} heures")
h.describe().round(1)
""")
lecture("uci")

md("## 1. Que consomme le chauffe-eau, et quand ?")
md("Le sous-compteur 3 mesure le **chauffe-eau et la climatisation**. C'est notre besoin d'eau chaude. Le reste de la conso (cuisine, buanderie, éclairage, etc.) ne sera pas piloté.")
code("""
hh = h.copy(); hh["heure"] = hh.index.hour; hh["mois"] = hh.index.month
jour = h.resample("D").sum() / 1000
print(f"Chauffe-eau : {jour.ecs_wh.median():.1f} kWh/jour en médiane (10e-90e centiles : {jour.ecs_wh.quantile(.1):.1f} - {jour.ecs_wh.quantile(.9):.1f})")
print(f"Part du chauffe-eau dans la conso totale : {jour.ecs_wh.sum() / jour.total_wh.sum():.0%}")

brut_sm3 = brut.Sub_metering_3.dropna()
print(f"Puissance du chauffe-eau quand il chauffe : {brut_sm3[brut_sm3 > 10].mode()[0] * 60 / 1000:.2f} kW (valeur la plus fréquente, en Wh/min x 60)")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
(hh.groupby("heure")[["ecs_wh", "autre_wh"]].mean() / 1000).plot(ax=axes[0], marker="o")
axes[0].axvspan(-0.5, 5.5, color="grey", alpha=.15); axes[0].axvspan(21.5, 23.5, color="grey", alpha=.15)
axes[0].set(title="Conso moyenne par heure (zones grises = heures creuses)", ylabel="kWh", xlabel="heure")
axes[0].legend(["chauffe-eau (+ clim)", "reste de la maison"])
(hh.groupby("mois")[["ecs_wh", "autre_wh"]].mean() * 24 / 1000).plot.bar(ax=axes[1], rot=0)
axes[1].set(title="Conso moyenne par jour selon le mois", ylabel="kWh/jour"); axes[1].legend(["chauffe-eau", "reste"])
plt.tight_layout(); plt.show()
""")
lecture("ecs")

md("## 2. Le solaire : combien, et quand ?")
code("""
pv = pvgis_local()
an = pv.pv_wh.resample("YE").sum() / 1000
print("Production annuelle (kWh) :", {d.year: int(v) for d, v in an.round(0).items()})
print(f"Soit {an.mean() / 3:.0f} kWh par kWc et par an en moyenne")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
(pv.pv_wh.groupby(pv.index.month).sum() / 1000 / 10).plot.bar(ax=axes[0], rot=0, color="#dd8452")
axes[0].set(title="Production moyenne par mois (3 kWc, 2014-2023)", ylabel="kWh", xlabel="mois")
prof = pv.assign(h=pv.index.hour, saison=np.where(pv.index.month.isin([6, 7, 8]), "été", np.where(pv.index.month.isin([12, 1, 2]), "hiver", "mi-saison")))
(prof.groupby(["h", "saison"]).pv_wh.mean().unstack() / 1000).plot(ax=axes[1], marker=".")
axes[1].set(title="Production moyenne par heure", ylabel="kWh", xlabel="heure (locale)")
plt.tight_layout(); plt.show()
""")
lecture("pv")

md("## 3. Le tarif Tempo : couleurs et prix")
code("""
t = pd.read_csv(DATA / "tempo_couleurs.csv", parse_dates=["date"])
t["saison"] = [f"{d.year}-{d.year + 1}" if d.month >= 9 else f"{d.year - 1}-{d.year}" for d in t.date]
cpt = t[t.saison <= "2025-2026"].groupby("saison").couleur.value_counts().unstack()[["bleu", "blanc", "rouge"]]
display(cpt.T)
rouge = t[t.couleur == "rouge"]
print("Mois des jours rouges :", rouge.date.dt.month.value_counts().sort_index().to_dict())
print("Jours de la semaine des jours rouges (0 = lundi) :", rouge.date.dt.dayofweek.value_counts().sort_index().to_dict())

grille = pd.DataFrame({c: {"heures creuses (22h-6h)": PRIX[(c, True)], "heures pleines (6h-22h)": PRIX[(c, False)]} for c in ["bleu", "blanc", "rouge"]})
print("Grille TTC au 1er août 2026 (€/kWh)"); display(grille)
""")
lecture("tempo")

md("## 4. Le jeu horaire du simulateur")
md("""
La conso date de 2006-2010, le solaire et Tempo de 2014-2023. `prepare_data.py` **rejoue** chaque jour cible avec le jour UCI de même semaine ISO et de même jour de la semaine, en faisant tourner l'année source (2007, 2008, 2009). Une journée Tempo va de 6 h à 6 h : de 0 h à 6 h, c'est la couleur de la veille qui s'applique. La couleur du lendemain n'est connue qu'à partir de 11 h (publication RTE).
""")
code("""
df = pd.read_parquet(DATA / "dataset_horaire.parquet")
print(f"{len(df):,} heures, du {df.index[0]} au {df.index[-1]} ; valeurs manquantes : {int(df.isna().sum().sum())}")
display(df.groupby("saison")[["autre_kwh", "ecs_kwh", "pv_kwh"]].sum().round(0).astype(int))
print("Découpage : entraînement 2014-2015 à 2020-2021, validation 2021-2022, test 2022-2023")
df.head(3)
""")
lecture("jeu")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "01_donnees.ipynb")
print("ok", len(C), "cellules")
