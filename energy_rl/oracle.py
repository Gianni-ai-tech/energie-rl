"""Optimum théorique : la facture minimale possible si l'on connaissait tout le futur (programme linéaire).

Même physique que le simulateur (ballon, pertes, batterie, solaire, prix Tempo, pénalité d'eau froide),
mais la chauffe et la batterie sont continues et pilotées librement, sans sécurité ni modes.
C'est une borne inférieure : aucun pilote réel, qui ne connaît pas l'avenir, ne peut faire mieux.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from energy_env import Maison


def optimum(m: Maison, debut: int, fin: int, ballon0: float | None = None, soc0: float = 0.0):
    p = m.p
    T = fin - debut
    D = m.ecs[debut:fin].astype(float)
    A_ = m.autre[debut:fin].astype(float)
    PV = m.pv[debut:fin].astype(float)
    prix = m.prix[debut:fin].astype(float)
    E0 = p.ballon_kwh * 0.8 if ballon0 is None else ballon0
    k, eta = p.pertes_h, p.rendement
    # variables : h, bin, bout, g, x, E, S, u  (8 blocs de T)
    nb = 8
    idx = {n: i * T for i, n in enumerate(["h", "bin", "bout", "g", "x", "E", "S", "u"])}
    N = nb * T
    c = np.zeros(N)
    c[idx["g"]:idx["g"] + T] = prix
    c[idx["u"]:idx["u"] + T] = p.penalite_eur_kwh
    r = np.arange(T)
    rows, cols, vals, b_eq = [], [], [], []
    ligne = 0

    def ajoute(coefs, rhs):
        nonlocal ligne
        for (bloc, decal), v in coefs:
            t = r + decal
            ok = t >= 0
            rows.extend((ligne + r[ok]).tolist()); cols.extend((idx[bloc] + t[ok]).tolist()); vals.extend([v] * int(ok.sum()))
        b_eq.extend(rhs.tolist())
        ligne += T

    # ballon : E_t - (1-k)(E_{t-1} + h_t + u_t) = -(1-k) D_t   (E_{-1} = E0 dans le second membre)
    rhs = -(1 - k) * D.copy(); rhs[0] += (1 - k) * E0
    ajoute([(("E", 0), 1.0), (("E", -1), -(1 - k)), (("h", 0), -(1 - k)), (("u", 0), -(1 - k))], rhs)
    # batterie : S_t - S_{t-1} - eta bin_t + bout_t / eta = 0
    rhs = np.zeros(T); rhs[0] = soc0
    ajoute([(("S", 0), 1.0), (("S", -1), -1.0), (("bin", 0), -eta), (("bout", 0), 1 / eta)], rhs)
    # bilan électrique : h + bin - bout - g + x = PV - autre
    ajoute([(("h", 0), 1.0), (("bin", 0), 1.0), (("bout", 0), -1.0), (("g", 0), -1.0), (("x", 0), 1.0)], PV - A_)
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(ligne, N))
    # contrainte de capacité avant pertes : E_{t-1} + h_t - D_t + u_t <= Emax
    rows2, cols2, vals2 = [], [], []
    for bloc, decal, v in [("E", -1, 1.0), ("h", 0, 1.0), ("u", 0, 1.0)]:
        t = r + decal; ok = t >= 0
        rows2.extend(r[ok].tolist()); cols2.extend((idx[bloc] + t[ok]).tolist()); vals2.extend([v] * int(ok.sum()))
    A_ub = sp.csr_matrix((vals2, (rows2, cols2)), shape=(T, N))
    b_ub = p.ballon_kwh + D; b_ub[0] -= E0
    bornes = ([(0, p.ballon_kw)] * T + [(0, p.batt_kw)] * T + [(0, p.batt_kw)] * T + [(0, None)] * T + [(0, None)] * T
              + [(0, p.ballon_kwh)] * T + [(0, p.batt_kwh)] * T + [(0, None)] * T)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array(b_eq), bounds=bornes, method="highs")
    assert res.status == 0, res.message
    z = res.x
    g = z[idx["g"]:idx["g"] + T]; u = z[idx["u"]:idx["u"] + T]
    return {"facture_eur": round(float((g * prix).sum()), 1), "eau_froide_kwh": round(float(u.sum()), 2),
            "achat_kwh": round(float(g.sum()), 0)}


if __name__ == "__main__":
    import time
    from energy_env import charger_donnees
    df = charger_donnees(); m = Maison(df)
    for s in ["2021-2022", "2022-2023"]:
        pos = np.where(df.saison == s)[0]
        t0 = time.time()
        print(s, optimum(m, int(pos[0]), int(pos[-1]) + 1), f"{time.time() - t0:.0f} s")
