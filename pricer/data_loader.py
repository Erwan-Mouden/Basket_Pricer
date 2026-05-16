from __future__ import annotations
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .curves import RateCurve, VolCurve

EXCEL_ORIGIN = datetime(1899, 12, 30)


def excel_date_to_datetime(excel_num):

    return EXCEL_ORIGIN + timedelta(days=int(excel_num))

#Fonction pour charger les données CAC40
def load_prices(path):

    df = pd.read_excel(path, sheet_name="Feuil1")

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.index.name = "date"

    # Nettoyage des noms de colonnes (espaces éventuels)
    df.columns = df.columns.str.strip()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    df = df.sort_index()

    return df


#Fonction pour charger les données de la courbe de taux
def load_rate_curve(path):

    df = pd.read_excel(path, sheet_name="TAUX")
    df.columns = df.columns.str.strip()

    times = []
    rates = []

    for _, row in df.iterrows():
        tenor = str(row["Tenor"]).strip().upper()
        yield_pct = float(str(row["Yield"]).replace(",", "."))

        # Conversion du tenor en années
        if tenor.endswith("M"):
            t = float(tenor[:-1]) / 12.0
        elif tenor.endswith("Y"):
            t = float(tenor[:-1])
        else:
            continue

        times.append(t)
        rates.append(yield_pct / 100.0)   # % → décimal

    return RateCurve(times, rates)


#Fonction pour charger la courbe de vol
def load_vol_curves(path, as_of):

    df = pd.read_excel(path, sheet_name="VOL_LONG")
    df.columns = df.columns.str.strip()

    # Conversion des dates Excel en datetime
    df["Expiry"] = pd.to_datetime(df["Expiry"], errors="coerce")

    vol_curves = {}

    for ticker, group in df.groupby("Ticker"):
        ticker = str(ticker).strip()
        times = []
        vols  = []

        for _, row in group.iterrows():
            expiry  = row["Expiry"]
            vol_pct = float(row["ATMVolPct"])

            # Maturité en années depuis as_of
            days = (expiry - as_of).days
            if days <= 0:
                continue        # on ignore les expiries passées

            t   = days / 365.0
            vol = vol_pct / 100.0   # % → décimal

            times.append(t)
            vols.append(vol)

        if len(times) >= 2:
            vol_curves[ticker] = VolCurve(times, vols)

    return vol_curves


def load_corr_matrix(prices, tickers):

    # On garde uniquement les tickers du panier
    df = prices[tickers].dropna()

    # Log-returns journaliers : ln(S_t / S_{t-1})
    log_rets = np.log(df / df.shift(1)).dropna()

    # Matrice de corrélation
    corr = log_rets.corr().values

    return corr