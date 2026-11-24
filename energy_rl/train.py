"""Entraîne l'agent PPO sur les saisons 2014-2015 à 2020-2021, choisit le meilleur modèle sur 2021-2022.

Usage : python train.py [pas_total] [nom]
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from energy_env import Maison, MaisonEnv, charger_donnees, simuler, bilan

torch.set_num_threads(1)
SAISONS_TRAIN = [f"{y}-{y+1}" for y in range(2014, 2021)]
SAISON_VALID = "2021-2022"
OUT = Path(__file__).resolve().parent / "modeles"


def bornes(df, saison):
    pos = np.where(df.saison == saison)[0]
    return int(pos[0]), int(pos[-1]) + 1


class ValidationCallback(BaseCallback):
    def __init__(self, m, a, b, tous_les, nom):
        super().__init__()
        self.m, self.a, self.b, self.tous_les, self.nom = m, a, b, tous_les, nom
        self.meilleur, self.histo = 1e9, []

    def _on_step(self):
        if self.n_calls % self.tous_les == 0:
            r = simuler(self.m, self.a, self.b, self.model)
            bl = bilan(r, self.m)
            score = bl["facture_eur"] + self.m.p.penalite_eur_kwh * bl["eau_froide_kwh"]
            self.histo.append({"pas": self.num_timesteps, **bl, "score": round(score, 1)})
            print(self.histo[-1], flush=True)
            if score < self.meilleur:
                self.meilleur = score
                self.model.save(OUT / f"{self.nom}_meilleur")
        return True


def main(pas_total=1_000_000, nom="ppo"):
    OUT.mkdir(exist_ok=True)
    df = charger_donnees()
    m = Maison(df)
    train_pos = np.where(df.saison.isin(SAISONS_TRAIN))[0]
    debuts = train_pos[train_pos < train_pos[-1] - 168]
    env = make_vec_env(lambda: MaisonEnv(m, debuts, duree=168), n_envs=4, seed=0)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=256, n_epochs=10,
                gamma=0.995, gae_lambda=0.95, ent_coef=0.01, clip_range=0.2,
                policy_kwargs=dict(net_arch=[128, 128]), seed=0, verbose=0)
    a, b = bornes(df, SAISON_VALID)
    cb = ValidationCallback(m, a, b, tous_les=50_000 // 4, nom=nom)
    t0 = time.time()
    model.learn(total_timesteps=pas_total, callback=cb)
    model.save(OUT / f"{nom}_final")
    (OUT / f"{nom}_historique.json").write_text(json.dumps(cb.histo, indent=1))
    print(f"durée {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000, sys.argv[2] if len(sys.argv) > 2 else "ppo")
