"""Construit le jeu de données horaire du simulateur (sept. 2014 -> août 2023).

Sources (dossier data/) :
- household_power_consumption.txt : conso minute d'une maison de Sceaux, 2006-2010 (UCI, CC BY 4.0)
- pvgis_sceaux_3kwc_2014_2023.csv : production horaire d'un 3 kWc à Sceaux (PVGIS, horodatage UTC)
- tempo_couleurs.csv : couleur Tempo de chaque jour depuis le 01/09/2014 (Jour Tempo / RTE)

Choix :
- La conso 2006-2010 est rejouée sur le calendrier 2014-2023 : chaque jour cible prend le jour UCI
  de même semaine ISO et de même jour de semaine, dans une année UCI qui tourne (2007, 2008, 2009).
- Le sous-compteur 3 (chauffe-eau + clim) sert de besoin d'eau chaude ; le reste est la conso non pilotable.
- Une journée Tempo va de 6 h à 6 h le lendemain : de 0 h à 6 h, c'est la couleur de la veille qui s'applique.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
START, END = "2014-09-01 06:00", "2023-08-31 23:00"


def uci_hourly() -> pd.DataFrame:
    df = pd.read_csv(DATA / "household_power_consumption.txt", sep=";", na_values="?", low_memory=False)
    ts = pd.to_datetime(df.Date + " " + df.Time, format="%d/%m/%Y %H:%M:%S")
    df = df.set_index(ts)
    total = df.Global_active_power * 1000 / 60  # Wh par minute
    h = pd.DataFrame({"total_wh": total, "ecs_wh": df.Sub_metering_3}).resample("h").sum(min_count=45)
    # trous : même heure de la semaine précédente, puis interpolation
    h = h.fillna(h.shift(168)).interpolate(limit_direction="both")
    h["autre_wh"] = (h.total_wh - h.ecs_wh).clip(lower=0)
    return h


def map_consumption(target_index: pd.DatetimeIndex, h: pd.DataFrame) -> pd.DataFrame:
    days = pd.DatetimeIndex(sorted(set(target_index.normalize())))
    src_rows = []
    for d in days:
        season = d.year - 2014 + (1 if d.month >= 9 else 0)  # 2014-15 -> 1
        uci_year = 2007 + (season % 3)
        iso = d.isocalendar()
        week = min(int(iso.week), 52)
        src = pd.Timestamp.fromisocalendar(uci_year, week, int(iso.weekday))
        src_rows.append((d, src))
    out = []
    for d, src in src_rows:
        block = h.loc[src: src + pd.Timedelta(hours=23)]
        if len(block) < 24:  # bord des données UCI
            block = h.loc[src - pd.Timedelta(days=7): src - pd.Timedelta(days=7) + pd.Timedelta(hours=23)]
        b = block.iloc[:24].copy()
        b.index = pd.date_range(d, periods=24, freq="h")
        b["jour_source"] = src.date()
        out.append(b)
    return pd.concat(out).loc[target_index]


def pvgis_local() -> pd.Series:
    raw = pd.read_csv(DATA / "pvgis_sceaux_3kwc_2014_2023.csv", skiprows=10, nrows=87648)
    ts = pd.to_datetime(raw.time, format="%Y%m%d:%H%M").dt.floor("h")
    s = pd.DataFrame({"pv_wh": raw.P.values, "t2m": raw.T2m.values}, index=ts)
    s.index = s.index.tz_localize("UTC").tz_convert("Europe/Paris").tz_localize(None)
    s = s[~s.index.duplicated()]  # heure dupliquée au passage à l'heure d'hiver
    full = pd.date_range(s.index.min().ceil("D"), s.index.max().floor("D") - pd.Timedelta(hours=1), freq="h")
    return s.reindex(full).interpolate(limit=2)


def tempo_by_hour(index: pd.DatetimeIndex) -> pd.Series:
    t = pd.read_csv(DATA / "tempo_couleurs.csv", parse_dates=["date"]).set_index("date").couleur
    jour_tempo = (index - pd.Timedelta(hours=6)).normalize()  # 0h-6h -> couleur de la veille
    return pd.Series(t.reindex(jour_tempo).values, index=index)


def main() -> pd.DataFrame:
    idx = pd.date_range(START, END, freq="h")
    h = uci_hourly()
    cons = map_consumption(idx, h)
    pv = pvgis_local().reindex(idx)
    df = pd.DataFrame(index=idx)
    df["autre_kwh"] = cons.autre_wh / 1000
    df["ecs_kwh"] = cons.ecs_wh / 1000
    df["pv_kwh"] = pv.pv_wh.fillna(0) / 1000
    df["t2m"] = pv.t2m.interpolate(limit_direction="both")
    df["tempo"] = tempo_by_hour(idx)
    # couleur du lendemain, connue à partir de 11 h (publication RTE vers 10h30-11h)
    jt = (idx - pd.Timedelta(hours=6)).normalize()
    t = pd.read_csv(DATA / "tempo_couleurs.csv", parse_dates=["date"]).set_index("date").couleur
    lendemain = pd.Series(t.reindex(jt + pd.Timedelta(days=1)).values, index=idx)
    connu = (idx.hour >= 11) | (idx.hour < 6)
    df["tempo_demain"] = lendemain.where(connu, "inconnu")
    df["hc"] = (idx.hour >= 22) | (idx.hour < 6)
    df["saison"] = [f"{y}-{y+1}" if m >= 9 else f"{y-1}-{y}" for y, m in zip(idx.year, idx.month)]
    df["jour_source_uci"] = cons.jour_source.values
    assert df.drop(columns=["jour_source_uci"]).isna().sum().sum() == 0, df.isna().sum()
    df.index.name = "heure"
    df.to_parquet(DATA / "dataset_horaire.parquet")
    return df


if __name__ == "__main__":
    d = main()
    print(d.shape)
    print(d.groupby("saison")[["autre_kwh", "ecs_kwh", "pv_kwh"]].sum().round(0))
    print(d.groupby("saison").tempo.value_counts().unstack().div(24).round(1))
