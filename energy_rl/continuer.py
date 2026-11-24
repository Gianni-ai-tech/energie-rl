"""Prolonge l'entraînement PPO d'un modèle existant (même environnement, même validation).

Usage : python continuer.py <nom> <pas_en_plus>
Le meilleur modèle (sur 2021-2022) continue d'écraser <nom>_meilleur seulement s'il bat le meilleur score déjà obtenu.
"""
import sys
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from energy_env import Maison, MaisonEnv, charger_donnees
from train import SAISONS_TRAIN, SAISON_VALID, OUT, bornes, ValidationCallback

torch.set_num_threads(2)


def main(nom, pas):
    df = charger_donnees(); m = Maison(df)
    train_pos = np.where(df.saison.isin(SAISONS_TRAIN))[0]
    debuts = train_pos[train_pos < train_pos[-1] - 168]
    env = make_vec_env(lambda: MaisonEnv(m, debuts, duree=168), n_envs=4, seed=7)
    model = PPO.load(OUT / f"{nom}_final", env=env, device="cpu")
    histo = json.loads((OUT / f"{nom}_historique.json").read_text())
    a, b = bornes(df, SAISON_VALID)
    cb = ValidationCallback(m, a, b, tous_les=50_000 // 4, nom=nom)
    cb.histo, cb.meilleur = histo, min(h["score"] for h in histo)
    model.learn(total_timesteps=pas, callback=cb, reset_num_timesteps=False)
    model.save(OUT / f"{nom}_final")
    (OUT / f"{nom}_historique.json").write_text(json.dumps(cb.histo, indent=1))
    print("meilleur score", cb.meilleur)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
