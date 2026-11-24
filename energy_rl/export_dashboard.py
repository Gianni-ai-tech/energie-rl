"""Précalcule les simulations de la saison test pour le tableau de bord (dashboard/traces_test.csv, bilan_test.csv).

Usage : python export_dashboard.py [nom_du_modele]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from energy_env import Maison, charger_donnees, simuler, bilan, REGLES
from oracle import optimum

ICI = Path(__file__).resolve().parent


def main(modele="ppo_v4"):
    df = charger_donnees(); m = Maison(df)
    pos = np.where(df.saison == "2022-2023")[0]; a, b = int(pos[0]), int(pos[-1]) + 1
    agent = PPO.load(ICI / "modeles" / f"{modele}_meilleur", device="cpu")
    pilotes = {**REGLES, "agent": agent}
    traces, bilans = [], {}
    for nom, pol in pilotes.items():
        r = simuler(m, a, b, pol)
        bilans[nom] = bilan(r, m)
        d = df.iloc[a:b]
        traces.append(pd.DataFrame({
            "heure": r.index, "pilote": nom, "tempo": d.tempo.values, "prix": d.prix.values,
            "pv_kwh": d.pv_kwh.values.round(3), "ecs_kwh": d.ecs_kwh.values.round(3),
            "achat": r.achat.values.round(3), "cout": r.cout.values.round(4), "ballon": r.ballon.values.round(3),
            "soc": r.soc.values.round(3), "manque": r.manque.values.round(3), "action": r.action.values}))
    bl = pd.DataFrame(bilans).T
    opt = optimum(m, a, b)
    bl.loc["optimum théorique", ["facture_eur", "achat_kwh", "eau_froide_kwh"]] = [opt["facture_eur"], opt["achat_kwh"], opt["eau_froide_kwh"]]
    (ICI / "dashboard").mkdir(exist_ok=True)
    pd.concat(traces).to_csv(ICI / "dashboard" / "traces_test.csv", index=False)
    bl.drop(index="optimum théorique").to_csv(ICI / "dashboard" / "bilan_test.csv")
    bl.to_csv(ICI / "resultats_test.csv")
    print(bl)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ppo_v4")
