"""Agent v3 : imitation de la règle experte, puis amélioration par PPO.

1. On rejoue la règle experte sur les saisons d'entraînement et on enregistre (observation, action, retour actualisé).
2. Le réseau de politique apprend à reproduire l'action de l'expert (classification), le réseau de valeur
   apprend le retour actualisé (régression) : l'agent part du niveau de l'expert au lieu de partir de zéro.
3. PPO améliore ensuite cette politique avec un pas d'apprentissage prudent, pour ne pas tout désapprendre.
Le meilleur modèle est choisi sur la saison 2021-2022 ; la saison 2022-2023 reste intacte pour le test.

Usage : python train_v3.py [pas_ppo] [nom]
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from energy_env import Maison, MaisonEnv, charger_donnees, simuler, regle_experte
from train import SAISONS_TRAIN, SAISON_VALID, OUT, bornes, ValidationCallback

torch.set_num_threads(2)


def demonstrations(m: Maison, df, gamma: float):
    obs, act, ret = [], [], []
    for s in SAISONS_TRAIN:
        a, b = bornes(df, s)
        r = simuler(m, a, b, regle_experte)
        ballon = np.r_[m.p.ballon_kwh * 0.8, r.ballon.to_numpy()[1:]]  # état avant chaque décision
        o = np.stack([m.observation(t, bl, sc) for t, bl, sc in
                      zip(range(a, b), r.ballon.to_numpy(), np.r_[0.0, r.soc.to_numpy()[:-1]])])
        rec = -(r.cout.to_numpy() + m.p.penalite_eur_kwh * r.manque.to_numpy())
        g = np.zeros_like(rec)
        acc = 0.0
        for i in range(len(rec) - 1, -1, -1):
            acc = rec[i] + gamma * acc
            g[i] = acc
        obs.append(o); act.append(r.action.to_numpy()); ret.append(g)
    return (torch.tensor(np.concatenate(obs), dtype=torch.float32),
            torch.tensor(np.concatenate(act), dtype=torch.long),
            torch.tensor(np.concatenate(ret), dtype=torch.float32))


def imitation(model: PPO, obs, act, ret, epoques=30, lot=512):
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    n = len(obs)
    for ep in range(epoques):
        perm = torch.randperm(n)
        pertes = []
        for i in range(0, n, lot):
            j = perm[i:i + lot]
            dist = pol.get_distribution(obs[j])
            logits = dist.distribution.logits
            perte_pi = F.cross_entropy(logits, act[j])
            v = pol.predict_values(obs[j]).squeeze(-1)
            perte_v = F.mse_loss(v, ret[j])
            perte = perte_pi + 0.05 * perte_v
            opt.zero_grad(); perte.backward(); opt.step()
            pertes.append((perte_pi.item(), perte_v.item()))
        if ep % 5 == 4 or ep == epoques - 1:
            with torch.no_grad():
                acc = (pol.get_distribution(obs).distribution.logits.argmax(1) == act).float().mean().item()
            pp, pv = np.mean(pertes, axis=0)
            print(f"imitation époque {ep + 1} : perte politique {pp:.3f}, perte valeur {pv:.2f}, accord avec l'expert {acc:.1%}", flush=True)


def main(pas_ppo=2_000_000, nom="ppo_v3", lr=1e-4, ent=0.001, clip=0.1, epochs=5):
    OUT.mkdir(exist_ok=True)
    df = charger_donnees()
    m = Maison(df)
    train_pos = np.where(df.saison.isin(SAISONS_TRAIN))[0]
    debuts = train_pos[train_pos < train_pos[-1] - 168]
    env = make_vec_env(lambda: MaisonEnv(m, debuts, duree=168), n_envs=4, seed=1)
    gamma = 0.995
    model = PPO("MlpPolicy", env, learning_rate=lr, n_steps=1024, batch_size=256, n_epochs=epochs,
                gamma=gamma, gae_lambda=0.95, ent_coef=ent, clip_range=clip,
                policy_kwargs=dict(net_arch=[128, 128]), seed=1, verbose=0)
    t0 = time.time()
    obs, act, ret = demonstrations(m, df, gamma)
    print(f"{len(obs):,} démonstrations de l'expert ({time.time() - t0:.0f} s)", flush=True)
    imitation(model, obs, act, ret)
    a, b = bornes(df, SAISON_VALID)
    cb = ValidationCallback(m, a, b, tous_les=50_000 // 4, nom=nom)
    cb.model = model
    from energy_env import bilan
    bl = bilan(simuler(m, a, b, model), m)
    sc = bl["facture_eur"] + m.p.penalite_eur_kwh * bl["eau_froide_kwh"]
    cb.histo.append({"pas": 0, **bl, "score": round(sc, 1)}); cb.meilleur = sc
    model.save(OUT / f"{nom}_meilleur")
    print("après imitation :", cb.histo[-1], flush=True)
    model.learn(total_timesteps=pas_ppo, callback=cb)
    model.save(OUT / f"{nom}_final")
    (OUT / f"{nom}_historique.json").write_text(json.dumps(cb.histo, indent=1))
    print(f"durée {time.time() - t0:.0f} s")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 2_000_000, a[1] if len(a) > 1 else "ppo_v3",
         *(float(x) for x in a[2:5]), *(int(x) for x in a[5:6]))
