from __future__ import annotations
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from pricer.data_loader import load_prices, load_rate_curve, load_vol_curves, load_corr_matrix
from pricer.models import Asset, Basket, OptionSpec, OptionType
from pricer.market_model import TermStructureModel
from pricer.pricers import monte_carlo_price

st.set_page_config(
    page_title="Basket Option Pricer",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = Path("data")

@st.cache_data
def get_prices():
    return load_prices(DATA_DIR / "data_CAC40.xlsx")


@st.cache_data
def get_rate_curve_points():
    """Retourne les points de la courbe de taux pour le graphique."""
    import pandas as pd
    df = pd.read_excel(DATA_DIR / "vol_impli.xlsx", sheet_name="TAUX")
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def get_vol_data():
    """Retourne les données brutes de vol pour les graphiques."""
    df = pd.read_excel(DATA_DIR / "vol_impli.xlsx", sheet_name="VOL_LONG")
    df.columns = df.columns.str.strip()
    df["Expiry"] = pd.to_datetime(df["Expiry"], errors="coerce")
    return df


st.title("📈 Basket Option Pricer")
st.caption("TermStructure Model (H2) — volatilités implicites ATM + courbe de taux")

prices     = get_prices()
as_of      = prices.index[-1].to_pydatetime()
all_tickers = list(prices.columns)

st.markdown(f"**As of :** {as_of.date()}  |  **{len(prices)} jours** de données  |  **{len(all_tickers)} sous-jacents**")
st.divider()


#Perf histo base 100
st.subheader("📊 Performance historique — base 100")

base100 = prices / prices.iloc[0] * 100

fig_perf = go.Figure()
for ticker in all_tickers:
    fig_perf.add_trace(go.Scatter(
        x=base100.index,
        y=base100[ticker],
        name=ticker,
        mode="lines",
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f}<extra>" + ticker + "</extra>",
    ))

fig_perf.update_layout(
    height=450,
    xaxis_title="Date",
    yaxis_title="Performance (base 100)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_perf, use_container_width=True)
st.divider()


# Matrice de correl histo
st.subheader("🔗 Matrice de corrélation historique")

corr_full = load_corr_matrix(prices, all_tickers)

short_names = [t.replace(" FP Equity", "").replace(" Equity", "") for t in all_tickers]

fig_corr = go.Figure(data=go.Heatmap(
    z=corr_full,
    x=short_names,
    y=short_names,
    colorscale="RdBu",
    zmin=-1, zmax=1,
    text=np.round(corr_full, 2),
    texttemplate="%{text}",
    textfont=dict(size=11),
    hoverongaps=False,
))
fig_corr.update_layout(
    height=420,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_corr, use_container_width=True)
st.divider()


#Courbe de taux
st.subheader("💹 Courbe de taux zéro-coupon")

taux_df = get_rate_curve_points()

def tenor_to_years(tenor):
    tenor = str(tenor).strip().upper()
    if tenor.endswith("M"):
        return float(tenor[:-1]) / 12.0
    elif tenor.endswith("Y"):
        return float(tenor[:-1])
    return None

taux_df["T"] = taux_df["Tenor"].apply(tenor_to_years)
taux_df = taux_df.dropna(subset=["T"])

fig_taux = go.Figure()
fig_taux.add_trace(go.Scatter(
    x=taux_df["T"],
    y=taux_df["Yield"],
    mode="lines+markers",
    name="Taux zéro-coupon",
    line=dict(color="#1f77b4", width=2),
    marker=dict(size=7),
    hovertemplate="T=%{x:.2f}y<br>Yield=%{y:.3f}%<extra></extra>",
))
fig_taux.update_layout(
    height=350,
    xaxis_title="Maturité (années)",
    yaxis_title="Taux (%)",
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig_taux, use_container_width=True)
st.divider()


#Vol term structure
st.subheader("📉 Volatilités implicites ATM par terme")

vol_data = get_vol_data()
vol_data["T"] = (vol_data["Expiry"] - pd.Timestamp(as_of)).dt.days / 365.0
vol_data = vol_data[vol_data["T"] > 0]

fig_vol = go.Figure()
for ticker in all_tickers:
    sub = vol_data[vol_data["Ticker"] == ticker].sort_values("T")
    if sub.empty:
        continue
    short = ticker.replace(" FP Equity", "").replace(" Equity", "")
    fig_vol.add_trace(go.Scatter(
        x=sub["T"],
        y=sub["ATMVolPct"],
        mode="lines+markers",
        name=short,
        hovertemplate="T=%{x:.2f}y<br>Vol=%{y:.1f}%<extra>" + short + "</extra>",
    ))

fig_vol.update_layout(
    height=400,
    xaxis_title="Maturité (années)",
    yaxis_title="Vol implicite ATM (%)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0),
)
st.plotly_chart(fig_vol, use_container_width=True)
st.divider()


#Interface du pricer
st.subheader("🎯 Pricer")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Sélection du panier**")

    selected = st.multiselect(
        "Sous-jacents",
        options=all_tickers,
        default=all_tickers[:3],
        format_func=lambda t: t.replace(" FP Equity", "").replace(" Equity", ""),
    )

    if len(selected) < 2:
        st.warning("Sélectionnez au moins 2 sous-jacents.")
        st.stop()

    st.markdown("**Poids du panier**")
    weight_mode = st.radio("", ["Équipondéré", "Manuel"], horizontal=True)

    if weight_mode == "Équipondéré":
        weights = np.full(len(selected), 1.0 / len(selected))
        st.info(f"Poids : {round(1/len(selected)*100, 1)}% par actif")
    else:
        weights = []
        cols = st.columns(len(selected))
        for i, (ticker, col) in enumerate(zip(selected, cols)):
            short = ticker.replace(" FP Equity", "").replace(" Equity", "")
            w = col.number_input(short, min_value=0.0, max_value=1.0,
                                  value=round(1/len(selected), 2), step=0.01)
            weights.append(w)
        weights = np.array(weights)
        total = weights.sum()
        if abs(total - 1.0) > 0.01:
            st.warning(f"Les poids somment à {total:.2f} — ils devraient sommer à 1.0")

    q = st.number_input("Dividende q", min_value=0.0, max_value=0.20,
                         value=0.02, step=0.005, format="%.3f")

with col2:
    st.markdown("**Paramètres de l'option**")

    T = st.slider("Maturité (années)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    option_type = st.radio("Type", ["Call", "Put"], horizontal=True)

    spots = [float(prices[t].iloc[-1]) for t in selected]
    basket_spot_preview = float(np.dot(weights[:len(spots)], spots))

    K = st.number_input(
        f"Strike (spot panier ≈ {basket_spot_preview:.2f})",
        min_value=1.0,
        value=round(basket_spot_preview, 2),
        step=1.0,
    )

    n_paths = st.select_slider(
        "Nombre de simulations",
        options=[10_000, 50_000, 100_000, 200_000, 500_000],
        value=200_000,
    )

    seed = st.number_input("Seed", min_value=0, value=42, step=1)

st.markdown("---")
if st.button("🚀 Lancer le pricer", type="primary", use_container_width=True):

    with st.spinner("Calcul en cours..."):
        try:
            assets = [
                Asset(ticker=t, spot=float(prices[t].iloc[-1]), dividend_yield=q)
                for t in selected
            ]
            basket = Basket(assets=assets, weights=weights)

            rate_curve = load_rate_curve(DATA_DIR / "vol_impli.xlsx")
            vol_curves_all = load_vol_curves(DATA_DIR / "vol_impli.xlsx", as_of)
            vol_curves = {t: vol_curves_all[t] for t in selected}
            corr = load_corr_matrix(prices, selected)

            model = TermStructureModel(
                basket=basket,
                rate_curve=rate_curve,
                vol_curves=vol_curves,
                corr=corr,
            )

            opt = OptionSpec(
                option_type=OptionType.CALL if option_type == "Call" else OptionType.PUT,
                strike=K,
                maturity=T,
            )

            result = monte_carlo_price(model, opt, n_paths=n_paths, seed=int(seed))

            lo, hi = result.ic95

            st.success("Calcul terminé !")

            r1, r2, r3 = st.columns(3)
            r1.metric("Prix", f"{result.price:.4f}")
            r2.metric("Std Error", f"{result.std_error:.4f}")
            r3.metric("IC 95%", f"[{lo:.4f} ; {hi:.4f}]")

            with st.expander("Détails du panier"):
                detail = pd.DataFrame({
                    "Ticker": selected,
                    "Spot": [float(prices[t].iloc[-1]) for t in selected],
                    "Poids": weights,
                    "Dividende q": q,
                })
                st.dataframe(detail, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Erreur : {e}")