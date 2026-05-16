"""
models.py
Structures de données du pricer H2.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class OptionType(Enum):
    CALL = "Call"
    PUT  = "Put"


@dataclass
class Asset:
    ticker:         str
    spot:           float
    dividend_yield: float


@dataclass
class Basket:
    assets:  list[Asset]
    weights: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.assets)

    @property
    def spot(self) -> float:
        return float(sum(w * a.spot for w, a in zip(self.weights, self.assets)))


@dataclass
class OptionSpec:
    option_type: OptionType
    strike:      float
    maturity:    float


@dataclass
class PriceResult:
    price:    float
    variance: float

    @property
    def std_error(self):
        return math.sqrt(self.variance)

    @property
    def ic95(self):
        se = self.std_error
        return (self.price - 1.96 * se, self.price + 1.96 * se)

    def __str__(self) :
        lo, hi = self.ic95
        return (
            f"Prix      = {self.price:.6f}\n"
            f"Std Error = {self.std_error:.6f}\n"
            f"IC 95%    = [{lo:.6f} ; {hi:.6f}]"
        )