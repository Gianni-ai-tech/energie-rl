"""Tableau de bord : pilotage d'un ballon d'eau chaude et d'une batterie au tarif Tempo (saison test 2022-2023).

Les simulations sont précalculées (export_dashboard.py) : le tableau de bord rejoue les décisions heure par heure.
"""
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Pilotage Tempo par RL", layout="wide")

NOMS = {"thermostat": "Thermostat (maison réelle)", "heures_creuses": "Contacteur heures creuses",
        "experte": "Règle experte", "agent": "Agent RL"}
COUL = {"bleu": "#2f6fb0", "blanc": "#9aa5b1", "rouge": "#c0392b"}
AXE_H = alt.Axis(format="%H h", title=None)


def fr(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


@st.cache_data
def charger():
    tr = pd.read_csv("traces_test.csv", parse_dates=["heure"])
    bl = pd.read_csv("bilan_test.csv", index_col=0)
    return tr, bl


tr, bl = charger()

st.title("Pilotage d'un chauffe-eau et d'une batterie au tarif Tempo")
st.caption("Maison de Sceaux (données UCI / EDF R&D), 3 kWc solaire (PVGIS), couleurs Tempo réelles, "
           "grille TTC au 1er août 2026. Saison test sept. 2022 – août 2023, jamais vue par l'agent pendant l'apprentissage.")

# ---------- bilan annuel ----------
st.subheader("Sur toute la saison test")
cols = st.columns(len(bl))
ref = bl.loc["thermostat", "facture_eur"]
for c, (k, r) in zip(cols, bl.iterrows()):
    delta = r.facture_eur - ref
    c.metric(NOMS.get(k, k), f"{r.facture_eur:,.0f} €".replace(",", " "))
    if k == "thermostat":
        c.caption("référence")
    else:
        c.caption(f"{delta:+,.0f} € ({100 * delta / ref:+.1f} %) face au thermostat".replace(",", " "))
st.dataframe(bl.rename(index=NOMS).rename(columns={
    "facture_eur": "facture (€)", "achat_kwh": "achat réseau (kWh)", "rouge_hp_kwh": "achat en heures pleines rouges (kWh)",
    "autoconso_pct": "solaire autoconsommé (%)", "eau_froide_h": "heures d'eau froide", "eau_froide_kwh": "eau chaude manquante (kWh)"}),
    use_container_width=True)

# ---------- une journée ----------
st.subheader("Rejouer une journée")
jours = sorted(tr.heure.dt.normalize().unique())
c1, c2 = st.columns([1, 2])
jour = c1.date_input("Jour", value=pd.Timestamp("2022-12-12").date(), min_value=jours[0].date(), max_value=jours[-2].date())
pilotes = c2.multiselect("Pilotes à comparer", list(NOMS), default=["experte", "agent"], format_func=NOMS.get)
debut = pd.Timestamp(jour) + pd.Timedelta(hours=6)
fen = tr[(tr.heure >= debut) & (tr.heure < debut + pd.Timedelta(hours=24))]
if fen.empty:
    st.info("Pas de données pour ce jour.")
    st.stop()
info = fen[fen.pilote == fen.pilote.iloc[0]]
couleur = info.tempo.iloc[0]
st.markdown(f"Journée Tempo du **{jour:%d/%m/%Y}** (de 6 h à 6 h le lendemain) : jour **{couleur}**. "
            f"Solaire produit : {fr(info.pv_kwh.sum())} kWh, besoin d'eau chaude : {fr(info.ecs_kwh.sum())} kWh.")

prix = alt.Chart(info).mark_line(interpolate="step-after", color="black").encode(
    x=alt.X("heure:T", axis=AXE_H), y=alt.Y("prix:Q", title="€/kWh")).properties(height=140, title="Prix de l'heure")
st.altair_chart(prix, use_container_width=True)

sel = fen[fen.pilote.isin(pilotes)].copy()
sel["pilote"] = sel.pilote.map(NOMS)
g1 = alt.Chart(sel).mark_bar(opacity=.75).encode(
    x=alt.X("heure:T", axis=AXE_H), y=alt.Y("achat:Q", title="kWh"), color=alt.Color("pilote:N", title=None),
    xOffset="pilote:N", tooltip=["pilote", "heure", alt.Tooltip("achat:Q", format=".2f"), alt.Tooltip("cout:Q", format=".2f")]
).properties(height=220, title="Électricité achetée au réseau")
g2 = alt.Chart(sel).mark_line(interpolate="step-after").encode(
    x=alt.X("heure:T", axis=AXE_H), y=alt.Y("ballon:Q", title="kWh", scale=alt.Scale(domain=[0, 7])), color=alt.Color("pilote:N", title=None)
).properties(height=200, title="Énergie dans le ballon (300 L, 7 kWh utiles)")
g3 = alt.Chart(sel).mark_line(interpolate="step-after").encode(
    x=alt.X("heure:T", axis=AXE_H), y=alt.Y("soc:Q", title="kWh", scale=alt.Scale(domain=[0, 5])), color=alt.Color("pilote:N", title=None)
).properties(height=200, title="Charge de la batterie (5 kWh)")
st.altair_chart(g1, use_container_width=True)
cc1, cc2 = st.columns(2)
cc1.altair_chart(g2, use_container_width=True)
cc2.altair_chart(g3, use_container_width=True)

res = sel.groupby("pilote").agg(facture=("cout", "sum"), achat=("achat", "sum"), eau_froide=("manque", "sum")).round(2)
res.columns = ["coût de la journée (€)", "achat (kWh)", "eau chaude manquante (kWh)"]
st.dataframe(res, use_container_width=True)

st.caption("Hypothèses : ballon 300 L / 3 kW avec sécurité sous 1 kWh, batterie 5 kWh / 2,5 kW (rendement 95 % par sens), "
           "surplus solaire non rémunéré, prévision solaire et couleur du lendemain (dès 11 h) connues. "
           "Code : github.com/Gianni-ai-tech/energie-rl")
