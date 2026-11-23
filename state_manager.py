"""Mantém o problema inicial e confirma apenas alterações com ótimo calculado."""


from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from sensitivity import Change, change_rhs, prices, ranges, slacks
from simplex import Problem, Result, solve


@dataclass
class Session:
    problem: Problem
    pivot_rule: str = "bland"
    original: Result = field(init=False)
    current: Result = field(init=False)
    rhs: np.ndarray = field(init=False)
    original_ranges: np.ndarray | None = field(init=False, default=None)
    original_slacks: np.ndarray | None = field(init=False, default=None)
    original_prices: np.ndarray | None = field(init=False, default=None)
    history: list[Change] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.original = self.current = solve(self.problem, pivot_rule=self.pivot_rule)
        self.rhs = self.problem.b
        if self.current.status == "ótimo":
            self.original_prices = prices(self.problem, self.original)
            self.original_ranges = ranges(self.original, self.rhs)
            self.original_slacks = slacks(self.problem, self.original, self.rhs)

    def apply(self, new_rhs, *, operations=()):
        """Registra a tentativa e troca a solução vigente quando há novo ótimo."""
        if self.current.status != "ótimo":
            raise ValueError("Resolva um PPL com solução ótima antes de alterar o LD.")
        change = change_rhs(self.problem, self.current, self.rhs, new_rhs, operations=operations)
        self.history.append(change)
        if change.result.status == "ótimo":
            self.current = change.result
            self.rhs = change.new_rhs
        return change

    def restore(self, number=None):
        """Reativa o quadro inicial ou o resultado ótimo de uma entrada do histórico."""
        rhs, result = self.problem.b, self.original
        if number is not None:
            if not 1 <= number <= len(self.history):
                raise ValueError("Alteração inexistente no histórico.")
            entry = self.history[number - 1]
            rhs, result = entry.new_rhs, entry.result
        if result.status != "ótimo":
            raise ValueError("O estado selecionado não tem quadro ótimo para restaurar.")
        change = change_rhs(self.problem, self.current, self.rhs, rhs, restored=result)
        self.history.append(change)
        self.current = result
        self.rhs = change.new_rhs
        return change
