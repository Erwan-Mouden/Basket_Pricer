from __future__ import annotations
import math

import numpy as np

from .curves import RateCurve, VolCurve
from .models import Basket


class TermStructureModel:

    def __init__(
        self,
        basket:     Basket,
        rate_curve: RateCurve,
        vol_curves: dict[str, VolCurve],
        corr:       np.ndarray,
    ):
        self.basket     = basket
        self.rate_curve = rate_curve
        self.vol_curves = vol_curves
        self.corr       = np.asarray(corr, dtype=float)

    @property
    def dim(self):
        """Nombre d'actifs dans le panier."""
        return self.basket.dim

    def discount_factor(self, T):
        """
        DF(0, T) = exp(-integral_0^T r(t) dt)
        """
        return self.rate_curve.df(T)

    def log_mean(self, i, T: float):

        asset    = self.basket.assets[i]
        vc       = self.vol_curves[asset.ticker]

        int_r    = self.rate_curve.integral_r(T)           # ∫ r(t) dt
        int_sig2 = vc.integral_vol2(T)                     # ∫ sigma(t)^2 dt

        return math.log(asset.spot) + int_r - asset.dividend_yield * T - 0.5 * int_sig2

    def log_var(self, i, T):

        asset = self.basket.assets[i]
        return self.vol_curves[asset.ticker].integral_vol2(T)

    def log_cov(self, i, j, T):

        if i == j:
            return self.log_var(i, T)

        ai = self.basket.assets[i]
        aj = self.basket.assets[j]

        int_prod = self.vol_curves[ai.ticker].integral_vol_product(
            self.vol_curves[aj.ticker], T
        )

        return self.corr[i, j] * int_prod