"""Simplex primal em duas fases e reotimização pelo Simplex Dual."""


from __future__ import annotations
from dataclasses import dataclass, replace

import numpy as np

from quadro_simplex import Step, Tableau, TOL


@dataclass(frozen=True)
class Problem:
    c: np.ndarray
    a: np.ndarray
    signs: tuple
    b: np.ndarray
    sense: str = "max"

    def __post_init__(self):
        for name in ("c", "a", "b"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            if not np.isfinite(array).all():
                raise ValueError("Os coeficientes e LDs devem ser finitos.")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        signs = tuple(self.signs)
        object.__setattr__(self, "signs", signs)
        if (self.c.shape not in ((2,), (3,), (4,)) or len(signs) == 0
                or self.b.shape != (len(signs),)
                or self.a.shape != (len(signs), self.c.size)
                or any(sign not in ("≤", "≥", "=") for sign in signs)
                or self.sense not in ("max", "min")):
            raise ValueError("Dimensões, sinais ou sentido do PPL inválidos.")

    @property
    def direction(self):
        return 1 if self.sense == "max" else -1


@dataclass(frozen=True)
class Result:
    status: str
    tableau: Tableau
    steps: tuple
    fixed: tuple = ()
    matrix: np.ndarray | None = None

    @property
    def base_matrix(self):
        return self.matrix[:, list(self.tableau.basis)]

    @property
    def basic_names(self):
        return tuple(self.tableau.names[j] for j in self.tableau.basis)

    @property
    def nonbasic_names(self):
        return tuple(name for j, name in enumerate(self.tableau.names)
                     if j not in self.tableau.basis)

    def point(self, n):
        values = np.zeros(n)
        for j, value in zip(self.tableau.basis, self.tableau.rhs):
            if j < n:
                values[j] = value
        return values


def _run(tableau, phase, dual=False, fixed=(), *, pivot_rule="bland"):
    """Percorre os quadros e registra o critério, as razões e o motivo da parada."""
    steps = []
    visited = set()
    status = "limite numérico"
    for _ in range(10_000):
        if tableau.basis in visited:
            if dual or pivot_rule == "bland":
                break
            # Bland permite continuar após uma repetição no critério do maior ganho.
            pivot_rule = "bland"
            visited.clear()
        visited.add(tableau.basis)
        with np.errstate(over="ignore", invalid="ignore"):
            reduced = tableau.reduced
            finite_objective = np.isfinite(tableau.z)
        if not finite_objective or not all(np.isfinite(values).all() for values in (
                reduced, tableau.rhs, tableau.coefficients, tableau.inverse)):
            break
        nonbasic = [j for j in range(len(tableau.names)) if j not in tableau.basis]
        if dual:
            rows = [i for i, b in enumerate(tableau.rhs)
                    if i not in fixed and b < -TOL]
            if not rows:
                status = "ótimo"
                break
            row = min(rows, key=lambda i: tableau.basis[i])
            ratios = [(j, max(0, -reduced[j]) / -tableau.coefficients[row, j])
                      for j in nonbasic if tableau.coefficients[row, j] < -TOL]
            if not ratios:
                status = "inviável"
                break
            column, _ = min(ratios, key=lambda item: (item[1], item[0]))
        else:
            columns = [j for j in nonbasic if reduced[j] > TOL]
            if not columns:
                status = "ótimo"
                break
            column = (min(columns, key=lambda j: (-reduced[j], j))
                      if pivot_rule == "professor" else min(columns))
            ratios = [(i, max(0, b) / tableau.coefficients[i, column])
                      for i, b in enumerate(tableau.rhs)
                      if tableau.coefficients[i, column] > TOL]
            if not ratios:
                status = "ilimitado"
                break
            row, _ = min(ratios, key=lambda item: (
                item[1], item[0] if pivot_rule == "professor" else tableau.basis[item[0]],
            ))
        steps.append(Step(phase, tableau, row, column, tuple(ratios), pivot_rule=pivot_rule))
        with np.errstate(over="ignore", invalid="ignore"):
            tableau = tableau.pivot(row, column)
    steps.append(Step(phase, tableau, note=f"Resultado da etapa · {status}"))
    return Result(status, tableau, tuple(steps))


def solve(problem, *, pivot_rule="bland"):
    """Normaliza o PPL, elimina artificiais e resolve o objetivo original por pivôs."""
    if pivot_rule not in ("bland", "professor"):
        raise ValueError("Critério de pivoteamento inválido.")
    m, n = problem.a.shape
    signs = np.where(problem.b < 0, -1., 1.)
    columns = list((signs[:, None] * problem.a).T)
    names = [f"x{j + 1}" for j in range(n)]
    artificial = []
    basis = []
    unit = np.eye(m)
    for i, (sign, direction) in enumerate(zip(problem.signs, signs)):
        if direction < 0:
            sign = {"≤": "≥", "≥": "≤", "=": "="}[sign]
        if sign != "=":
            names.append(f"{'f' if sign == '≤' else 'e'}{i + 1}")
            columns.append(unit[:, i] if sign == "≤" else -unit[:, i])
        if sign != "≤":
            artificial.append(len(names))
            names.append(f"a{i + 1}")
            columns.append(unit[:, i])
        basis.append(len(names) - 1)

    normal = np.array(columns).T
    original_matrix = signs[:, None] * normal
    # A identidade com os sinais das linhas acompanha os pivôs até virar B⁻¹.
    tableau = Tableau(tuple(names), normal, abs(problem.b), np.diag(signs),
                      tuple(basis), np.zeros(len(names)))
    steps = []
    if artificial:
        costs = tableau.costs.copy()
        costs[artificial] = -1
        phase_one = _run(replace(tableau, costs=costs), "Fase 1", pivot_rule=pivot_rule)
        steps.extend(phase_one.steps)
        tableau = phase_one.tableau
        if phase_one.status != "ótimo":
            return replace(phase_one, status="limite numérico")
        if tableau.z < -TOL:
            return replace(phase_one, status="inviável")

    fixed = []
    for row, basic in enumerate(tableau.basis):
        if basic not in artificial:
            continue
        candidates = [j for j in range(len(names))
                      if j not in artificial and j not in tableau.basis
                      and abs(tableau.coefficients[row, j]) > TOL]
        if candidates:
            column = min(candidates)
            steps.append(Step("Transição", tableau, row, column,
                              note="Retirada de artificial básica com valor zero."))
            tableau = tableau.pivot(row, column)
        else:
            fixed.append(row)

    # Em linhas dependentes, a artificial básica funciona como marcador fixo em zero.
    retained = [j for j in range(len(names))
                if j not in artificial or j in tableau.basis]
    costs = np.zeros(len(retained))
    costs[:n] = problem.direction * problem.c
    coefficients = tableau.coefficients[:, retained].copy()
    for row in fixed:
        coefficients[row] = 0
        coefficients[row, retained.index(tableau.basis[row])] = 1
    tableau = replace(
        tableau, names=tuple(names[j] for j in retained), costs=costs,
        coefficients=coefficients,
        basis=tuple(retained.index(j) for j in tableau.basis),
    )
    result = _run(tableau, "Fase 2", fixed=tuple(fixed), pivot_rule=pivot_rule)
    return replace(result, steps=tuple(steps) + result.steps,
                   fixed=tuple(fixed), matrix=original_matrix[:, retained])


def dual_reoptimize(reference, tableau):
    """Restaura a viabilidade primal no quadro de referência com LD atualizado."""
    if any(abs(tableau.rhs[i]) > TOL for i in reference.fixed):
        step = Step("Simplex Dual", tableau,
                    note="Os LDs violam uma dependência entre as restrições.")
        return replace(reference, status="inviável", tableau=tableau, steps=(step,))
    result = _run(tableau, "Simplex Dual", dual=True, fixed=reference.fixed)
    return replace(result, fixed=reference.fixed, matrix=reference.matrix)
