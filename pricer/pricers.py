"""
pricers.py
Monte Carlo avec control variate géométrique.

Fonctions publiques :
  - monte_carlo_price : prix d'une option sur panier via MC + CV géométrique

Fonctions internes :
  - _geo_basket_price : prix analytique du panier géométrique (control variate)
"""

from __future__ import annotations
import math

import numpy as np
from scipy.stats import norm

from .models import Basket, OptionSpec, OptionType, PriceResult
from .market_model import TermStructureModel


def monte_carlo_price(
    model:   TermStructureModel,
    opt:     OptionSpec,
    n_paths: int = 200_000,
    seed:    int = 42,
) -> PriceResult:
    """
    Prix d'une option européenne sur panier arithmétique.
    Monte Carlo + control variate géométrique.

    Algorithme
    ----------
    1. Cholesky sur la matrice de corrélation des log-prix
    2. Simulation de n_paths trajectoires corrélées
    3. Calcul des payoffs arithmétique (Y) et géométrique (X)
    4. Estimation de beta = Cov(Y,X) / Var(X)
    5. Estimateur ajusté : Y + beta * (geo_analytique - X)

    Paramètres
    ----------
    model   : TermStructureModel
    opt     : spécification de l'option
    n_paths : nombre de simulations
    seed    : graine du générateur aléatoire

    Retourne
    --------
    PriceResult(price, variance)
    """
    n  = model.dim
    b  = model.basket
    T  = opt.maturity
    K  = opt.strike
    df = model.discount_factor(T)

    # ── Étape 1 : écarts-types et matrice de corrélation des log-prix ──
    # std_i = sqrt(Var[ln Si(T)]) = sqrt(∫ sigma_i(t)^2 dt)
    std = np.array([
        math.sqrt(max(model.log_var(i, T), 1e-16))
        for i in range(n)
    ])

    # Matrice de corrélation des log-prix terminaux
    # corr_log[i,j] = Cov[ln Si, ln Sj] / (std_i * std_j)
    corr_log = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov        = model.log_cov(i, j, T)
            denom      = std[i] * std[j]
            corr_log[i, j] = cov / denom if denom > 1e-16 else (1.0 if i == j else 0.0)

    # ── Étape 2 : décomposition de Cholesky ──
    # corr_log = L @ L.T
    # On utilise numpy qui est plus stable numériquement
    L = np.linalg.cholesky(corr_log)

    # ── Prix analytique du panier géométrique (control variate) ──
    geo_analytic = _geo_basket_price(model, opt)

    # ── Étape 3 : simulation ──
    rng = np.random.default_rng(seed)

    def _simulate():
        """
        Simule n_paths trajectoires et retourne (Y, X) :
        Y = payoff actualisé arithmétique
        X = payoff actualisé géométrique
        """
        # Tirage de gaussiennes indépendantes (n_paths x n)
        # puis corrélation via Cholesky : Z_corr = Z @ L.T
        Z      = rng.standard_normal((n_paths, n))
        Z_corr = Z @ L.T                             # (n_paths x n)

        # ln Si(T) = mean_i + std_i * Z_corr_i
        means = np.array([model.log_mean(i, T) for i in range(n)])
        ln_S  = means + std * Z_corr                 # (n_paths x n)
        S_T   = np.exp(ln_S)                         # (n_paths x n)

        # Panier arithmétique : B = sum(w_i * Si)
        B_arith = S_T @ b.weights                    # (n_paths,)

        # Panier géométrique : G = exp(sum(w_i * ln Si))
        B_geo = np.exp(ln_S @ b.weights)             # (n_paths,)

        # Payoffs
        if opt.option_type == OptionType.CALL:
            payoff_arith = np.maximum(B_arith - K, 0.0)
            payoff_geo   = np.maximum(B_geo   - K, 0.0)
        else:
            payoff_arith = np.maximum(K - B_arith, 0.0)
            payoff_geo   = np.maximum(K - B_geo,   0.0)

        Y = df * payoff_arith    # payoff actualisé arithmétique
        X = df * payoff_geo      # payoff actualisé géométrique

        return Y, X

    # ── Étape 4 : estimation de beta (passe 1) ──
    rng = np.random.default_rng(seed)
    Y1, X1 = _simulate()

    mean_Y = Y1.mean()
    mean_X = X1.mean()
    var_X  = ((X1 - mean_X) ** 2).mean()
    cov_XY = ((X1 - mean_X) * (Y1 - mean_Y)).mean()

    beta = cov_XY / var_X if var_X > 1e-18 else 0.0

    # ── Étape 5 : estimateur ajusté (passe 2, mêmes tirages) ──
    rng = np.random.default_rng(seed)
    Y2, X2 = _simulate()

    # Estimateur avec control variate
    adj = Y2 + beta * (geo_analytic - X2)

    price    = float(adj.mean())
    var_samp = float(((adj - price) ** 2).mean())
    var_est  = var_samp / n_paths

    return PriceResult(price=price, variance=var_est)


def _geo_basket_price(model: TermStructureModel, opt: OptionSpec) -> float:
    """
    Prix analytique de l'option sur panier géométrique.

    ln G_T = sum(w_i * ln Si(T)) est gaussien
    → formule Black-Scholes directe

    C'est le control variate : très corrélé au panier arithmétique
    mais avec un prix exact → réduit la variance du MC.
    """
    n  = model.dim
    b  = model.basket
    T  = opt.maturity
    K  = opt.strike
    df = model.discount_factor(T)

    # Moyenne du log du panier géométrique
    # m = sum(w_i * E[ln Si(T)])
    m = sum(b.weights[i] * model.log_mean(i, T) for i in range(n))

    # Variance du log du panier géométrique
    # v = sum_i sum_j w_i * w_j * Cov[ln Si, ln Sj]
    v = sum(
        b.weights[i] * b.weights[j] * model.log_cov(i, j, T)
        for i in range(n)
        for j in range(n)
    )
    v = max(v, 1e-16)
    s = math.sqrt(v)

    # E[G_T] = exp(m + 0.5 * v)
    EG = math.exp(m + 0.5 * v)

    # Black-Scholes sur G_T
    d1 = (math.log(EG / K) + 0.5 * v) / s
    d2 = d1 - s

    call = df * (EG * norm.cdf(d1) - K * norm.cdf(d2))

    if opt.option_type == OptionType.CALL:
        return call

    # Put via parité call-put
    return call - df * (EG - K)