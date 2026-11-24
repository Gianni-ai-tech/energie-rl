"""Simulateur horaire d'une maison au tarif Tempo : ballon d'eau chaude + batterie + 3 kWc solaires.

Chaque heure, le pilote choisit :
- ballon : 0 = ne pas chauffer, 1 = chauffer (3 kW) ;
- batterie : 0 = autoconsommation (charge le surplus solaire, décharge si déficit),
             1 = charger (surplus solaire puis réseau, 2,5 kW max),
             2 = conserver (charge le surplus solaire, ne décharge pas).
Action Gymnasium = 3 * ballon + batterie (6 actions).

Hypothèses (toutes dans PARAMS, pour le cahier des charges) : voir README.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset_horaire.parquet"

# Grille Tempo TTC au 1er août 2026 (délibération CRE du 15/07/2026), €/kWh
PRIX = {("bleu", True): 0.1356, ("bleu", False): 0.1654,
        ("blanc", True): 0.1536, ("blanc", False): 0.1921,
        ("rouge", True): 0.1615, ("rouge", False): 0.7295}
COULEURS = ["bleu", "blanc", "rouge"]
DEMAIN = ["bleu", "blanc", "rouge", "inconnu"]


@dataclass
class Params:
    ballon_kwh: float = 7.0        # énergie utile stockée (300 L, de 45 à 65 °C)
    ballon_kw: float = 3.0         # résistance
    pertes_h: float = 0.005        # pertes horaires en plus, proportionnelles à l'énergie stockée
    batt_kwh: float = 5.0
    batt_kw: float = 2.5
    rendement: float = 0.95        # par sens (aller-retour ~0,90)
    penalite_eur_kwh: float = 3.0  # coût d'une eau trop froide, par kWh manquant
    horizon_pv: int = 6            # prévision solaire supposée parfaite sur 6 h
    securite_kwh: float = 1.0      # sécurité thermostat : chauffe forcée sous ce niveau, quel que soit le pilote


def charger_donnees() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    df["prix"] = [PRIX[(c, h)] for c, h in zip(df.tempo, df.hc)]
    return df


class Maison:
    """Physique de la maison, partagée par l'environnement RL et les règles de référence."""

    def __init__(self, df: pd.DataFrame, p: Params = Params()):
        self.p = p
        self.df = df
        self.autre = df.autre_kwh.to_numpy(np.float32)
        self.ecs = df.ecs_kwh.to_numpy(np.float32)
        self.pv = df.pv_kwh.to_numpy(np.float32)
        self.prix = df.prix.to_numpy(np.float32)
        self.hc = df.hc.to_numpy(bool)
        self.t2m = df.t2m.to_numpy(np.float32)
        self.coul = df.tempo.map({c: i for i, c in enumerate(COULEURS)}).to_numpy(np.int8)
        self.dem = df.tempo_demain.map({c: i for i, c in enumerate(DEMAIN)}).to_numpy(np.int8)
        idx = df.index
        self.heure = idx.hour.to_numpy()
        self.jsem = idx.dayofweek.to_numpy()
        self.mois = idx.month.to_numpy()
        cs = np.concatenate([[0.0], np.cumsum(self.pv)])
        n = len(self.pv)
        fin = np.minimum(np.arange(n) + 1 + p.horizon_pv, n)
        self.pv_futur = (cs[fin] - cs[np.minimum(np.arange(n) + 1, n)]).astype(np.float32)
        fin24 = np.minimum(np.arange(n) + 1 + 24, n)
        self.pv_24h = (cs[fin24] - cs[np.minimum(np.arange(n) + 1, n)]).astype(np.float32)  # prévision météo J+1 supposée parfaite
        # besoin d'eau chaude moyen à la même heure sur les 7 jours précédents (connu du pilote)
        e = pd.Series(self.ecs)
        self.ecs_habitude = sum(e.shift(24 * k) for k in range(1, 8)).div(7).bfill().to_numpy(np.float32)
        self._prix_anticipes()

    def _prix_anticipes(self, horizon: int = 24):
        """Prix des prochaines heures tels que le pilote peut les connaître à l'heure t.
        Couleur du jour Tempo en cours : connue. Lendemain : connu après 11 h, sinon supposé bleu."""
        n = len(self.prix)
        jt = ((self.df.index - pd.Timedelta(hours=6)).normalize() - pd.Timestamp("2014-01-01")).days.to_numpy()
        bleu = np.where(self.hc, PRIX[("bleu", True)], PRIX[("bleu", False)]).astype(np.float32)
        futur = np.zeros((n, horizon), np.float32)
        for k in range(1, horizon + 1):
            j = np.minimum(np.arange(n) + k, n - 1)
            meme_jour = jt[j] == jt
            connu = meme_jour | (self.dem != 3)
            futur[:, k - 1] = np.where(connu, self.prix[j], bleu[j])
        self.prix_12 = futur[:, :12].mean(1)
        self.prix_max16 = futur[:, :16].max(1)
        self.prix_suivant = futur[:, 0]
        rouge = futur > 0.5
        self.h_avant_rouge = np.where(rouge.any(1), rouge.argmax(1) + 1, horizon + 24).astype(np.float32)
        h = self.heure
        self.h_avant_hc = np.where(self.hc, 0, 22 - h).astype(np.float32)
        self.autre_prec = np.concatenate([[self.autre[0]], self.autre[:-1]]).astype(np.float32)

    def pas(self, t: int, ballon: float, batt_soc: float, chauffe: int, mode_batt: int):
        """Avance d'une heure. Renvoie (ballon, soc, infos)."""
        p = self.p
        # ballon (la sécurité du thermostat force la chauffe si le ballon est presque vide)
        if ballon < p.securite_kwh:
            chauffe = 1
        chauffe_kwh = p.ballon_kw if chauffe else 0.0
        chauffe_kwh = min(chauffe_kwh, p.ballon_kwh - ballon + self.ecs[t])
        ballon = ballon + chauffe_kwh - self.ecs[t]
        manque = max(0.0, -ballon)
        ballon = max(0.0, ballon) * (1 - p.pertes_h)
        # bilan électrique avant batterie
        charge = self.autre[t] + chauffe_kwh
        net = charge - self.pv[t]  # > 0 : déficit, < 0 : surplus
        place = (p.batt_kwh - batt_soc) / p.rendement
        dispo = batt_soc * p.rendement
        batt_in = batt_out = 0.0
        if net < 0:  # surplus solaire : toujours stocké si possible
            batt_in = min(-net, p.batt_kw, place)
        if mode_batt == 0 and net > 0:
            batt_out = min(net, p.batt_kw, dispo)
        if mode_batt == 1:
            batt_in = min(p.batt_kw, place)  # surplus puis réseau
        batt_soc = batt_soc + batt_in * p.rendement - batt_out / p.rendement
        achat = max(0.0, net + batt_in - batt_out)
        perdu = max(0.0, -(net + batt_in - batt_out))  # surplus non stocké, injecté gratuitement
        cout = achat * self.prix[t]
        info = dict(achat=achat, cout=cout, manque=manque, chauffe=chauffe_kwh,
                    pv_auto=self.pv[t] - perdu, perdu=perdu, batt_in=batt_in, batt_out=batt_out)
        return ballon, min(max(batt_soc, 0.0), p.batt_kwh), info

    def observation(self, t: int, ballon: float, soc: float) -> np.ndarray:
        p = self.p
        h, j, m = self.heure[t], self.jsem[t], self.mois[t]
        o = np.zeros(27, np.float32)
        o[0:2] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
        o[2:4] = np.sin(2 * np.pi * j / 7), np.cos(2 * np.pi * j / 7)
        o[4:6] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
        o[6 + self.coul[t]] = 1
        o[9 + self.dem[t]] = 1
        o[13] = self.hc[t]
        o[14] = self.prix[t] / 0.7295
        o[15] = ballon / p.ballon_kwh
        o[16] = soc / p.batt_kwh
        o[17] = self.pv[t] / 3.0
        o[18] = self.pv_futur[t] / 12.0
        o[19] = self.ecs_habitude[t] / 1.2
        o[20] = self.prix_suivant[t] / 0.7295
        o[21] = self.prix_12[t] / 0.7295
        o[22] = self.prix_max16[t] / 0.7295
        o[23] = min(self.h_avant_rouge[t], 48) / 48
        o[24] = self.h_avant_hc[t] / 16
        o[25] = self.autre_prec[t] / 3.0
        o[26] = self.pv_24h[t] / 20.0
        return o


class MaisonEnv(gym.Env):
    """Épisodes de `duree` heures tirés au hasard dans les heures autorisées."""

    def __init__(self, maison: Maison, debuts: np.ndarray, duree: int = 168, seed: int | None = None):
        self.m, self.debuts, self.duree = maison, debuts, duree
        self.observation_space = spaces.Box(-1.5, 1.5, shape=(27,), dtype=np.float32)
        self.action_space = spaces.Discrete(6)
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        opts = options or {}
        self.t0 = int(opts.get("debut", self.rng.choice(self.debuts)))
        self.t = self.t0
        self.ballon = float(opts.get("ballon", self.rng.uniform(0.3, 1.0) * self.m.p.ballon_kwh))
        self.soc = float(opts.get("soc", self.rng.uniform(0, 1) * self.m.p.batt_kwh))
        return self.m.observation(self.t, self.ballon, self.soc), {}

    def step(self, action):
        chauffe, mode = divmod(int(action), 3)
        self.ballon, self.soc, info = self.m.pas(self.t, self.ballon, self.soc, chauffe, mode)
        recompense = -(info["cout"] + self.m.p.penalite_eur_kwh * info["manque"])
        self.t += 1
        fini = self.t - self.t0 >= self.duree or self.t >= len(self.m.prix)
        obs = self.m.observation(min(self.t, len(self.m.prix) - 1), self.ballon, self.soc)
        return obs, float(recompense), False, fini, info


# ---------- pilotages de référence ----------

def regle_thermostat(m: Maison, t, ballon, soc):
    """Ce que fait la maison réelle : le ballon se réchauffe dès qu'on tire de l'eau."""
    return 3 * int(ballon < m.p.ballon_kwh - 0.5) + 0


def regle_heures_creuses(m: Maison, t, ballon, soc):
    """Pilotage classique : contacteur heures creuses, relance de secours si le ballon est presque vide."""
    chauffe = (m.hc[t] and ballon < m.p.ballon_kwh) or ballon < 1.0
    return 3 * int(chauffe) + 0


def regle_experte(m: Maison, t, ballon, soc):
    """Règle écrite à la main avec le tarif en tête : la référence que l'agent doit battre."""
    rouge_hp = m.coul[t] == 2 and not m.hc[t]
    surplus = m.pv[t] - m.autre[t]
    if rouge_hp:
        chauffe = ballon < 0.5
    elif surplus > 1.5:
        chauffe = ballon < m.p.ballon_kwh
    else:
        chauffe = (m.hc[t] and ballon < m.p.ballon_kwh) or ballon < 1.0
    # batterie : remplie la nuit (heures creuses) avant un jour blanc ou rouge, gardée pour les heures pleines
    veille_cher = m.hc[t] and m.dem[t] in (1, 2) and m.heure[t] >= 22 or (m.hc[t] and m.heure[t] < 6 and m.dem[t] in (1, 2))
    if veille_cher:
        mode = 1
    elif m.hc[t]:
        mode = 2
    else:
        mode = 0
    return 3 * int(chauffe) + mode


REGLES = {"thermostat": regle_thermostat, "heures_creuses": regle_heures_creuses, "experte": regle_experte}


def simuler(m: Maison, debut: int, fin: int, politique, ballon0=None, soc0=0.0) -> pd.DataFrame:
    """Rejoue une politique (fonction (m,t,ballon,soc)->action, ou modèle SB3) sur [debut, fin)."""
    ballon = m.p.ballon_kwh * 0.8 if ballon0 is None else ballon0
    soc = soc0
    lignes = []
    for t in range(debut, fin):
        if callable(politique):
            a = politique(m, t, ballon, soc)
        else:
            a, _ = politique.predict(m.observation(t, ballon, soc), deterministic=True)
        chauffe, mode = divmod(int(a), 3)
        avant = ballon
        ballon, soc, info = m.pas(t, ballon, soc, chauffe, mode)
        info.update(t=t, action=int(a), ballon=avant, soc=soc)
        lignes.append(info)
    r = pd.DataFrame(lignes)
    r.index = m.df.index[debut:fin]
    return r


def bilan(r: pd.DataFrame, m: Maison) -> dict:
    d = m.df.loc[r.index]
    rouge_hp = (d.tempo == "rouge") & ~d.hc
    pv = d.pv_kwh.sum()
    return {
        "facture_eur": round(float(r.cout.sum()), 1),
        "achat_kwh": round(float(r.achat.sum()), 0),
        "rouge_hp_kwh": round(float(r.achat[rouge_hp].sum()), 1),
        "autoconso_pct": round(float(100 * r.pv_auto.sum() / pv), 1) if pv else 0.0,
        "eau_froide_h": int((r.manque > 1e-6).sum()),
        "eau_froide_kwh": round(float(r.manque.sum()), 2),
    }
