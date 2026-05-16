from __future__ import annotations
import math
import numpy as np

class RateCurve:

    def __init__(self, times: list[float], rates: list[float]):
        # on trie par maturite croissante
        pairs = sorted(zip(times, rates), key=lambda p: p[0])
        self._t      = np.array([p[0] for p in pairs], dtype=float)
        self._rates  = np.array([p[1] for p in pairs], dtype=float)
        # DF(t) = exp(-r * t)
        self._df= np.exp(-self._rates * self._t)
        self._log_df = np.log(self._df)

    def df(self, t: float): #test si interpolation ou extrapolation

        if t <= 0:
            return 1.0
        if t <= self._t[0]:
            return self._interp(t, 0, 1)
        if t >= self._t[-1]:
            return self._interp(t, len(self._t) - 2, len(self._t) - 1)

        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        idx = max(0, min(idx, len(self._t) - 2))
        return self._interp(t, idx, idx + 1)

    def integral_r(self, t: float):
        return -math.log(self.df(t))
 
    def _interp(self, t: float, i0: int, i1: int): #gère l'interpolation
        t0, t1 = self._t[i0], self._t[i1]
        if abs(t1 - t0) < 1e-12:
            return float(self._df[i0])
        w = (t - t0) / (t1 - t0)
        log_df = (1.0 - w) * self._log_df[i0] + w * self._log_df[i1]
        return math.exp(log_df) 


class VolCurve:
 
    def __init__(self, times: list[float], vols: list[float]):
        pairs = sorted(zip(times, vols), key=lambda p: p[0])
        self._t = np.array([p[0] for p in pairs], dtype=float)
        self._v = np.array([p[1] for p in pairs], dtype=float)
 
    def vol(self, t: float) -> float:
        if t <= self._t[0]:
            return float(self._v[0])
        if t >= self._t[-1]:
            return float(self._v[-1])
 
        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        idx = max(0, min(idx, len(self._t) - 2))
        t0, t1 = self._t[idx], self._t[idx + 1]
        w = (t - t0) / (t1 - t0)
        return float((1.0 - w) * self._v[idx] + w * self._v[idx + 1])
 
    def integral_vol2(self, T: float, steps: int = 200):
        if T <= 0:
            return 0.0
 
        dt = T / steps
        t0, v0 = 0.0, self.vol(0.0)
        total = 0.0
 
        for k in range(1, steps + 1):
            t1 = k * dt
            v1 = self.vol(t1)
            total += 0.5 * (v0 * v0 + v1 * v1) * dt
            t0, v0 = t1, v1
 
        return total
 
    def integral_vol_product(self, other: VolCurve, T: float, steps: int = 200):
        if T <= 0:
            return 0.0
 
        dt = T / steps
        t0 = 0.0
        va0, vb0 = self.vol(0.0), other.vol(0.0)
        total = 0.0
 
        for k in range(1, steps + 1):
            t1 = k * dt
            va1 = self.vol(t1)
            vb1 = other.vol(t1)
            total += 0.5 * (va0 * vb0 + va1 * vb1) * dt
            t0 = t1
            va0, vb0 = va1, vb1
 
        return total
 