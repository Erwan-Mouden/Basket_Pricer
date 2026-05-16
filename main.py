"""
main.py
Point d'entrée interactif du Basket Pricer H2.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from pricer.data_loader import load_prices, load_rate_curve, load_vol_curves, load_corr_matrix
from pricer.models import Asset, Basket, OptionSpec, OptionType
from pricer.market_model import TermStructureModel
from pricer.pricers import monte_carlo_price

DATA_DIR = Path("data")


def parse_float(prompt):
    while True:
        try:
            return float(input(prompt).strip().replace(",", "."))
        except ValueError:
            print("  Valeur invalide, réessayez.")


def parse_int(prompt, default):
    raw = input(prompt).strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def main():
    print("=" * 55)
    print("  Basket Option Pricer — TermStructure Model (H2)")
    print("=" * 55)

    # ── 1. Chargement des prix ──
    prices = load_prices(DATA_DIR / "data_CAC40.xlsx")
    as_of  = prices.index[-1].to_pydatetime()
    all_tickers = list(prices.columns)

    print(f"\n  Données chargées : {len(prices)} dates")
    print(f"  As of            : {as_of.date()}")
    print(f"\n  Tickers disponibles :")
    for i, t in enumerate(all_tickers, 1):
        print(f"    {i:>3}. {t}")

    # ── 2. Sélection des tickers ──
    print("\n  Entrez les numéros séparés par virgules (ex: 1,3,5) ou ALL :")
    while True:
        raw = input("  > ").strip().upper()
        if raw == "ALL":
            tickers = all_tickers
            break
        try:
            indices = [int(x.strip()) for x in raw.split(",")]
            tickers = [all_tickers[i - 1] for i in indices if 1 <= i <= len(all_tickers)]
            tickers = list(dict.fromkeys(tickers))
            if len(tickers) >= 2:
                break
            print("  Sélectionnez au moins 2 tickers.")
        except (ValueError, IndexError):
            print("  Format invalide.")

    print(f"\n  Panier : {', '.join(tickers)}")

    # ── 3. Poids ──
    print("\n  Entrez les poids séparés par virgules (ex: 0.3,0.3,0.4)")
    print("  ou appuyez sur Entrée pour équipondérer :")
    while True:
        raw = input("  > ").strip()
        if raw == "":
            weights = np.full(len(tickers), 1.0 / len(tickers))
            print(f"  Poids équipondérés : {round(1/len(tickers), 4)} chacun")
            break
        try:
            weights = np.array([float(x.strip().replace(",", ".")) for x in raw.split(",")])
            if len(weights) != len(tickers):
                print(f"  Il faut {len(tickers)} poids.")
                continue
            break
        except ValueError:
            print("  Format invalide.")

    # ── 4. Dividende ──
    q = parse_float("\n  Dividende q pour tous les actifs (ex: 0.02) : ")

    # ── 5. Construction du modèle ──
    assets = [
        Asset(
            ticker=t,
            spot=float(prices[t].iloc[-1]),
            dividend_yield=q,
        )
        for t in tickers
    ]
    basket = Basket(assets=assets, weights=weights)

    rate_curve  = load_rate_curve(DATA_DIR / "vol_impli.xlsx")
    vol_curves  = load_vol_curves(DATA_DIR / "vol_impli.xlsx", as_of)
    corr        = load_corr_matrix(prices, tickers)

    # Filtrer les vol_curves sur les tickers du panier
    vol_curves = {t: vol_curves[t] for t in tickers}

    model = TermStructureModel(
        basket=basket,
        rate_curve=rate_curve,
        vol_curves=vol_curves,
        corr=corr,
    )

    basket_spot = basket.spot
    print(f"\n  Spot du panier : {basket_spot:.4f}")

    # ── 6. Option ──
    T = parse_float("\n  Maturité en années (ex: 1.0) : ")

    print("\n  Type d'option :")
    print("    1. Call")
    print("    2. Put")
    while True:
        choix = input("  > ").strip()
        if choix == "1":
            option_type = OptionType.CALL
            break
        elif choix == "2":
            option_type = OptionType.PUT
            break
        print("  Entrez 1 ou 2.")

    print(f"\n  Strike (Entrée pour ATM = {basket_spot:.4f}) :")
    raw = input("  > ").strip().replace(",", ".")
    K   = float(raw) if raw else basket_spot

    opt = OptionSpec(option_type=option_type, strike=K, maturity=T)
    print(f"\n  Option : {option_type.value}  Strike={K:.4f}  T={T:.2f}y")

    # ── 7. Menu ──
    menu = """
  ┌─────────────────────────────────────┐
  │  1  Pricer (Monte Carlo + CV géo)   │
  │  0  Quitter                         │
  └─────────────────────────────────────┘"""

    while True:
        print(f"\n  Panier spot ≈ {basket_spot:.4f} | K={K:.4f} | T={T:.2f}y | {len(tickers)} actifs")
        print(menu)
        choix = input("  Choix : ").strip()

        if choix == "0":
            print("\n  Au revoir.\n")
            break

        elif choix == "1":
            n_paths = parse_int("  Nb simulations [200000] : ", 200_000)
            seed    = parse_int("  Seed [42] : ", 42)

            print("\n  Calcul en cours...")
            result = monte_carlo_price(model, opt, n_paths=n_paths, seed=seed)
            print(f"\n  {result}")

        else:
            print("  Choix invalide.")


if __name__ == "__main__":
    main()