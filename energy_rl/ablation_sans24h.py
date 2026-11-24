"""Contrôle : même entraînement que ppo_v4 (imitation + PPO, mêmes réglages, graine 1, 2 millions d'heures),
mais la prévision solaire à 24 h (observation 26) est masquée (mise à 0) à l'entraînement et en validation.
Sert à vérifier si cette prévision explique le gain de l'agent.
Usage : python ablation_sans24h.py  -> modeles/ppo_v4_sans24h_*"""
import energy_env

_obs = energy_env.Maison.observation


def observation_sans24h(self, t, ballon, soc):
    o = _obs(self, t, ballon, soc)
    o[26] = 0.0
    return o


energy_env.Maison.observation = observation_sans24h

import train_v3  # noqa: E402

if __name__ == "__main__":
    train_v3.main(2_000_000, "ppo_v4_sans24h", 3e-4, 0.01, 0.2, 10)
