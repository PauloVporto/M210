"""Sensibilidade ao LD, expressa nas coordenadas do problema original."""


from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np

from metodo_simplex import Result, dual_reoptimize
from quadro_simplex import TOL


def prices(problem, result):
    """Calcula CbᵀB⁻¹ na escala da função objetivo original."""
    t = result.tableau
    return problem.direction * (t.cb @ t.inverse)


def ranges(result, rhs):
    """Intersecta as condições de cada linha quando apenas um LD varia."""
    t = result.tableau
    bounds = []
    for b, column in zip(rhs, t.inverse.T):
        lower, upper = -np.inf, np.inf
        for row, (value, factor) in enumerate(zip(t.rhs, column)):
            if abs(factor) <= TOL:
                continue
            limit = -max(0, value) / factor
            if row in result.fixed:
                lower = upper = 0.
                break
            if factor > 0:
                lower = max(lower, limit)
            else:
                upper = min(upper, limit)
        bounds.append((b + lower, b + upper))
    return np.array(bounds)


def slacks(problem, result, rhs):
    """Retorna folgas ou excessos na ordem das restrições originais."""
    residual = rhs - problem.a @ result.point(problem.c.size)
    return np.array([-value if sign == "≥" else value
                     for sign, value in zip(problem.signs, residual)])


@dataclass(frozen=True)
class Change:
    old_rhs: np.ndarray
    new_rhs: np.ndarray
    previous: Result
    result: Result
    shadow: np.ndarray
    trial: np.ndarray
    violations: tuple
    old_z: float
    new_z: float | None
    method: str
    previous_ranges: np.ndarray
    operations: tuple = ()

    @property
    def delta(self):
        return self.new_rhs - self.old_rhs


def change_rhs(problem, reference, old_rhs, new_rhs, *, restored=None, operations=()):
    """Verifica a proposta conjunta e reotimiza só se a base perder viabilidade."""
    new_rhs = np.array(new_rhs, dtype=float, copy=True)
    if new_rhs.shape != old_rhs.shape or not np.isfinite(new_rhs).all():
        raise ValueError("Informe um LD finito para cada restrição.")
    t = reference.tableau
    with np.errstate(over="ignore", invalid="ignore"):
        delta = new_rhs - old_rhs
        trial = t.rhs + t.inverse @ delta
    if not np.isfinite(trial).all() or not np.isfinite(delta).all():
        raise ArithmeticError("A variação ultrapassou a capacidade de cálculo. Revise a escala dos LDs.")
    candidate = replace(t, rhs=trial)
    shadow = prices(problem, reference)
    old_z = problem.direction * t.z
    violations = tuple(name for i, (name, value) in
                       enumerate(zip(reference.basic_names, trial))
                       if value < -TOL or (i in reference.fixed and value > TOL))
    if restored is not None:
        result = restored
        method = "Restauração"
        new_z = problem.direction * restored.tableau.z
    elif violations:
        result = dual_reoptimize(reference, candidate)
        method = "Simplex Dual"
        new_z = problem.direction * result.tableau.z if result.status == "ótimo" else None
    else:
        result = replace(reference, tableau=candidate, steps=())
        method = "Preço-sombra"
        with np.errstate(over="ignore", invalid="ignore"):
            new_z = float(old_z + shadow @ delta)
    if new_z is not None and not np.isfinite(new_z):
        raise ArithmeticError("O valor da função objetivo ultrapassou a capacidade de cálculo.")
    return Change(old_rhs, new_rhs, reference, result, shadow, trial,
                  violations, old_z, new_z, method, ranges(reference, old_rhs), tuple(operations))
