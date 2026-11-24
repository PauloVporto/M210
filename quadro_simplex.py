"""Operações de linha e registro dos quadros Simplex."""


from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np

TOL = 1e-9


@dataclass(frozen=True)
class Tableau:
    names: tuple
    coefficients: np.ndarray
    rhs: np.ndarray
    inverse: np.ndarray
    basis: tuple
    costs: np.ndarray

    @property
    def cb(self):
        return self.costs[list(self.basis)]

    @property
    def zj(self):
        return self.cb @ self.coefficients

    @property
    def reduced(self):
        return self.costs - self.zj

    @property
    def z(self):
        return float(self.cb @ self.rhs)

    def pivot(self, row, col):
        """Aplica Gauss–Jordan à coluna pivô e às colunas auxiliares de B⁻¹."""
        joined = np.column_stack((self.coefficients, self.rhs, self.inverse))
        factors = self.coefficients[:, col].copy()
        pivot_row = joined[row] / factors[row]
        # O broadcasting aplica o coeficiente de cada linha à mesma linha normalizada.
        joined -= factors[:, None] * pivot_row
        joined[row] = pivot_row
        joined[:, col] = 0
        joined[row, col] = 1
        width = len(self.names)
        basis = list(self.basis)
        basis[row] = col
        return replace(
            self, coefficients=joined[:, :width], rhs=joined[:, width],
            inverse=joined[:, width + 1:], basis=tuple(basis),
        )


@dataclass(frozen=True)
class Step:
    phase: str
    tableau: Tableau
    row: int | None = None
    column: int | None = None
    ratios: tuple = ()
    note: str = ""
    pivot_rule: str = "bland"
