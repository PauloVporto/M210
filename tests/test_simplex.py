import unittest
from unittest.mock import patch

import numpy as np

from sensitivity import prices, ranges, slacks
from simplex import Problem, solve
from state_manager import Session


def example():
    return Problem([3, 2], [[1, 1], [1, 0], [0, 1]], ("≤",) * 3, [4, 2, 3])


class SimplexTests(unittest.TestCase):
    def test_original_tableau_and_sensitivity(self):
        problem = example()
        result = solve(problem)
        self.assertEqual(result.status, "ótimo")
        self.assertEqual(result.tableau.z, 10)
        np.testing.assert_allclose(result.point(2), [2, 2])
        np.testing.assert_allclose(prices(problem, result), [2, 1, 0])
        np.testing.assert_allclose(slacks(problem, result, problem.b), [0, 0, 1])
        np.testing.assert_allclose(ranges(result, problem.b), [[2, 5], [1, 4], [2, np.inf]])
        np.testing.assert_allclose(result.tableau.inverse @ result.base_matrix, np.eye(3))
        for before, after in zip(result.steps, result.steps[1:]):
            pivoted = before.tableau.pivot(before.row, before.column)
            np.testing.assert_allclose(pivoted.coefficients, after.tableau.coefficients)
            np.testing.assert_allclose(pivoted.rhs, after.tableau.rhs)

    def test_minimization_two_phases(self):
        problem = Problem([2, 1, 3], [[1, 1, 0], [1, 0, 0], [0, 0, 1]],
                          ["≥", "≥", "="], [4, 1, 0], "min")
        session = Session(problem)
        self.assertEqual(session.original.status, "ótimo")
        np.testing.assert_allclose(session.original.point(3), [1, 3, 0])
        np.testing.assert_allclose(prices(problem, session.original), [1, 1, 3])
        self.assertEqual(session.apply([5, 1, 0]).new_z, 6)
        self.assertIn("Fase 1", [step.phase for step in session.original.steps])

    def test_negative_rhs_and_original_shadow_sign(self):
        problem = Problem([1, 0], [[-1, 0], [0, 1]], ["≥", "≤"], [-2, 1])
        session = Session(problem)
        np.testing.assert_allclose(prices(problem, session.current), [-1, 0])
        self.assertEqual(session.apply([-3, 1]).new_z, 3)

    def test_infeasible_and_unbounded(self):
        impossible = Problem([1, 0], [[1, 0], [1, 0]], ["≥", "≤"], [2, 1])
        unbounded = Problem([1, 0], [[0, 1]], ["≤"], [1])
        self.assertEqual(solve(impossible).status, "inviável")
        self.assertEqual(solve(unbounded).status, "ilimitado")

    def test_degenerate_ties_and_cycling_example(self):
        problem = Problem([10, -57, -9, -24],
                          [[.5, -5.5, -2.5, 9], [.5, -1.5, -.5, 1], [1, 0, 0, 0]],
                          ["≤"] * 3, [0, 0, 1])
        first, second = solve(problem), solve(problem)
        self.assertEqual(first.status, "ótimo")
        self.assertAlmostEqual(first.tableau.z, 1)
        self.assertEqual([(s.row, s.column) for s in first.steps],
                         [(s.row, s.column) for s in second.steps])
        professor = solve(problem, pivot_rule="professor")
        self.assertEqual(professor.status, "ótimo")
        self.assertAlmostEqual(professor.tableau.z, 1)
        self.assertEqual(professor.steps[0].pivot_rule, "professor")
        self.assertIn("bland", [step.pivot_rule for step in professor.steps if step.row is not None])


class SensitivityTests(unittest.TestCase):
    def test_successive_updates_reuse_tableau_without_any_solver(self):
        session = Session(example())
        original = session.original
        original_basic = original.tableau.rhs.copy()
        with (patch("simplex.solve", side_effect=AssertionError),
              patch("state_manager.solve", side_effect=AssertionError),
              patch("simplex._run", side_effect=AssertionError),
              patch("tableau.Tableau.pivot", side_effect=AssertionError)):
            first = session.apply([4.5, 2, 3])
            second = session.apply([4.5, 2.5, 3])
        self.assertEqual([first.new_z, second.new_z], [11, 11.5])
        self.assertIs(session.original, original)
        self.assertIs(second.result.tableau.inverse, original.tableau.inverse)
        self.assertEqual(second.method, "Preço-sombra")
        np.testing.assert_allclose(first.new_rhs, second.old_rhs)
        np.testing.assert_allclose(session.original.tableau.rhs, original_basic)
        np.testing.assert_allclose(session.problem.b, [4, 2, 3])
        np.testing.assert_allclose(first.result.point(2), [2, 2.5])
        self.assertEqual(len(session.history), 2)

    def test_joint_violation_uses_dual_and_never_extrapolates_z(self):
        session = Session(example())
        proposed = [5, 1, 3]
        low, high = session.original_ranges.T
        self.assertTrue(np.all(np.array(proposed) >= low))
        self.assertTrue(np.all(np.array(proposed) <= high))
        with (patch("simplex.solve", side_effect=AssertionError),
              patch("state_manager.solve", side_effect=AssertionError)):
            change = session.apply(proposed)
        self.assertTrue(change.violations)
        self.assertEqual(change.method, "Simplex Dual")
        self.assertNotEqual(change.new_z, change.old_z + change.shadow @ change.delta)
        self.assertEqual(change.new_z, 9)
        self.assertNotEqual(change.previous.tableau.basis, session.current.tableau.basis)
        np.testing.assert_allclose(prices(session.problem, session.current), [0, 3, 2])
        self.assertEqual(session.apply([5, 1, 2.5]).new_z, 8)

    def test_infeasible_update_preserves_committed_state_and_history(self):
        session = Session(example())
        failure = session.apply([4, -1, 3])
        self.assertIsNone(failure.new_z)
        self.assertEqual(failure.result.status, "inviável")
        self.assertIs(session.current, session.original)
        np.testing.assert_allclose(session.rhs, [4, 2, 3])
        self.assertEqual(session.apply([4, 3, 3]).new_z, 11)
        self.assertIs(session.history[0], failure)

    def test_three_confirmed_updates_from_request(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        with patch("simplex._run", side_effect=AssertionError("Pivoteamento inesperado")):
            for rhs, z, point in (
                ([110, 80], 2700, [30, 50]),
                ([110, 88], 2860, [22, 66]),
                ([105, 88], 2810, [17, 71]),
            ):
                change = session.apply(rhs)
                self.assertEqual(change.new_z, z)
                self.assertEqual(change.violations, ())
                self.assertEqual(change.method, "Preço-sombra")
                np.testing.assert_allclose(change.trial, point)
                np.testing.assert_allclose(change.result.point(2), point)
        np.testing.assert_allclose(session.history[2].old_rhs, [110, 88])

    def test_nonfinite_objective_cannot_commit_an_update(self):
        problem = Problem([1e200, 0], [[1, 0]], ["≤"], [1])
        session = Session(problem)
        with self.assertRaisesRegex(ArithmeticError, "função objetivo"):
            session.apply([1e200])
        self.assertIs(session.current, session.original)
        self.assertEqual(session.history, [])
        np.testing.assert_allclose(session.rhs, [1])

    def test_dependent_equalities_require_joint_consistency(self):
        problem = Problem([1, 0], [[1, 0], [2, 0]], ["=", "="], [1, 2])
        session = Session(problem)
        self.assertTrue(session.original.fixed)
        np.testing.assert_allclose(session.original_ranges, [[1, 1], [2, 2]])
        self.assertEqual(session.apply([2, 4]).new_z, 2)
        self.assertEqual(session.apply([3, 4]).result.status, "inviável")
        self.assertEqual(session.apply([1, 2]).new_z, 1)


if __name__ == "__main__":
    unittest.main()
