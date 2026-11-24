"""Robustesse de l'agent à une prévision solaire imparfaite (saisons validation et test).

L'agent a été entraîné avec une prévision parfaite du solaire des 6 et 24 prochaines heures
(observations 18 et 26). On le réévalue, sans le réentraîner, avec :
  - une prévision bruitée : la production de chaque jour prévu est multipliée par un facteur
    exp(N(0, sigma)), tiré une fois par jour (une journée entière plus ou moins ensoleillée
    que prévu), 5 tirages par niveau de bruit ;
  - la persistance : « demain sera comme aujourd'hui », c'est-à-dire aucune prévision météo.
Contrôle : l'agent ppo_v4_sans24h, entraîné exactement comme ppo_v4 mais sans la prévision à 24 h
(observation 26 à 0, voir ablation_sans24h.py), évalué sur les mêmes saisons.
La production réelle, la mesure de l'heure en cours et la règle experte (qui n'utilise pas
de prévision) ne changent pas.

Usage : python robustesse_prevision.py  -> resultats_robustesse.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from energy_env import Maison, bilan, charger_donnees, regle_experte, simuler

ICI = Path(__file__).parent
SIGMAS = [0.1, 0.25, 0.5]
TIRAGES = 5


def cumul_futur(serie: np.ndarray, h: int) -> np.ndarray:
    """Somme de serie sur les h heures qui suivent t (t+1 .. t+h), comme dans Maison."""
    n = len(serie)
    cs = np.concatenate([[0.0], np.cumsum(serie)])
    debut = np.minimum(np.arange(n) + 1, n)
    fin = np.minimum(np.arange(n) + 1 + h, n)
    return (cs[fin] - cs[debut]).astype(np.float32)


def prevision_bruitee(m: Maison, sigma: float, rng) -> None:
    jours = m.df.index.normalize()
    code, uniques = pd.factorize(jours)
    facteur = np.exp(rng.normal(0.0, sigma, len(uniques)))[code]
    pv_prevu = m.pv * facteur
    m.pv_futur = cumul_futur(pv_prevu, m.p.horizon_pv)
    m.pv_24h = cumul_futur(pv_prevu, 24)


def persistance(m: Maison) -> None:
    """Prévision = ce qui a été produit 24 h plus tôt sur les mêmes heures."""
    hier = np.concatenate([np.repeat(m.pv[:1], 24), m.pv[:-24]])
    m.pv_futur = cumul_futur(hier, m.p.horizon_pv)
    m.pv_24h = cumul_futur(hier, 24)


class MaisonSans24h(Maison):
    def observation(self, t, ballon, soc):
        o = super().observation(t, ballon, soc)
        o[26] = 0.0
        return o


def main():
    df = charger_donnees()
    agent = PPO.load(ICI / "modeles" / "ppo_v4_meilleur", device="cpu")
    controle = PPO.load(ICI / "modeles" / "ppo_v4_sans24h_meilleur", device="cpu")
    lignes = []
    for saison in ["2021-2022", "2022-2023"]:
        pos = np.where(df.saison == saison)[0]
        a, b = int(pos[0]), int(pos[-1]) + 1

        m = Maison(df)
        ref = bilan(simuler(m, a, b, regle_experte), m)
        lignes.append(dict(saison=saison, scenario="règle experte", sigma=np.nan, tirage=0, **ref))
        lignes.append(dict(saison=saison, scenario="agent, prévision parfaite", sigma=0.0, tirage=0,
                           **bilan(simuler(m, a, b, agent), m)))
        for s in SIGMAS:
            for k in range(TIRAGES):
                m = Maison(df)
                prevision_bruitee(m, s, np.random.default_rng(1000 * k + int(100 * s)))
                lignes.append(dict(saison=saison, scenario=f"agent, bruit {int(100 * s)} %", sigma=s, tirage=k,
                                   **bilan(simuler(m, a, b, agent), m)))
                print(saison, s, k, lignes[-1]["facture_eur"], flush=True)
        m = Maison(df)
        persistance(m)
        lignes.append(dict(saison=saison, scenario="agent, persistance (sans météo)", sigma=np.nan, tirage=0,
                           **bilan(simuler(m, a, b, agent), m)))
        print(saison, "persistance", lignes[-1]["facture_eur"], flush=True)
        m = MaisonSans24h(df)
        lignes.append(dict(saison=saison, scenario="contrôle : agent entraîné sans la prévision à 24 h", sigma=np.nan,
                           tirage=0, **bilan(simuler(m, a, b, controle), m)))
        print(saison, "contrôle", lignes[-1]["facture_eur"], flush=True)
    r = pd.DataFrame(lignes)
    r.to_csv(ICI / "resultats_robustesse.csv", index=False)
    print(r.groupby(["saison", "scenario"], sort=False)[["facture_eur", "achat_kwh", "autoconso_pct", "eau_froide_h"]]
          .agg(["mean", "min", "max"]).round(1).to_string())


if __name__ == "__main__":
    main()
