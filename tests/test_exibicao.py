"""Regressões para a apresentação manual e os exercícios da NP2."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

from exibicao import format_number, manual_rows
from relatorios import REPORTS, report_pdf
from contas_recursos import relative_percentage, resource_rhs
from metodo_simplex import Problem
from sessao import Session

APP = Path(__file__).resolve().parents[1] / "aplicacao.py"


def render(session):
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["session"] = session
    app.session_state["model_version"] = 1
    return app.run()


def displayed_text(app, tab):
    return "\n".join(item.value for item in app.tabs[tab].text)


class ManualPresentationTests(unittest.TestCase):
    def test_example_from_request_shows_conditions_and_lucro(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        app = render(session)
        self.assertFalse(app.exception)
        sensitivity = displayed_text(app, 2)
        self.assertIn("PS(R1) = 10  ·  Linha Z, coluna F1", sensitivity)
        self.assertIn("PS(R2) = 20  ·  Linha Z, coluna F2", sensitivity)
        self.assertIn("20 + Δ1 − Δ2 ≥ 0", sensitivity)
        self.assertIn("60 − Δ1 + 2Δ2 ≥ 0", sensitivity)
        app.multiselect[0].set_value([0]).run()
        field = next(item for item in app.number_input if item.label == "Valor de R1")
        field.set_value(110.)
        submit = next(item for item in app.button if item.label == "Calcular novo Z")
        with patch("metodo_simplex._run", side_effect=AssertionError("Pivoteamento inesperado")):
            submit.click().run()
        self.assertFalse(app.exception)
        text = displayed_text(app, 3)
        for line in (
            "20 + (10) − (0) = 30 ≥ 0  (OK)",
            "60 − (10) + 2(0) = 50 ≥ 0  (OK)",
            "ZNOVO = 2600 + 10Δ1 + 20Δ2",
            "ZNOVO = 2600 + 10(10) + 20(0)",
            "ZNOVO = 2700",
            "x1 = 30",
            "x2 = 50",
        ):
            self.assertIn(line, text)

    def test_np2_exercise_one_tableaux_and_items_c_and_d(self):
        session = Session(Problem([80, 70, 100, 16], [
            [1, 1, 1, 4], [0, 1, 1, 2], [3, 2, 4, 0],
        ], ["≤"] * 3, [250, 600, 500]), pivot_rule="professor")
        initial = session.original.steps[0].tableau
        rows = list(manual_rows(initial))
        self.assertEqual(rows[0], ["Z", -80, -70, -100, -16, 0, 0, 0, 0])
        self.assertEqual([r[0] for r in rows], ["Z", "F1", "F2", "F3"])
        self.assertEqual(session.original.tableau.z, 17500)
        np.testing.assert_allclose(session.original.point(4), [0, 250, 0, 0])
        steps = session.original.steps
        self.assertEqual([(s.row, s.column) for s in steps], [(2, 2), (0, 1), (None, None)])
        np.testing.assert_allclose(list(manual_rows(steps[1].tableau))[0][1:],
                                   [-5, -20, 0, -16, 0, 0, 25, 12500])
        np.testing.assert_allclose(list(manual_rows(steps[2].tableau))[0][1:],
                                   [5, 0, 0, 144, 40, 0, 15, 17500])
        np.testing.assert_allclose(session.original_prices, [40, 0, 15])
        change_c = session.apply([500, 600, 500])
        np.testing.assert_allclose(change_c.trial, [750, 100, -250])
        self.assertEqual(change_c.violations, ("x3",))
        self.assertEqual(change_c.method, "Simplex Dual")
        self.assertEqual(change_c.new_z, 18500)
        session.restore()
        change = session.apply([225, 540, 450])
        np.testing.assert_allclose(change.trial, [225, 315, 0])
        self.assertEqual(change.method, "Preço-sombra")
        self.assertFalse(change.violations)
        app = render(session)
        self.assertFalse(app.exception)
        self.assertIn("ZNOVO = 15750", displayed_text(app, 3))

    def test_np2_exercise_two_prices_and_item_c(self):
        session = Session(Problem([20, 40], [[4, 15], [6, 30], [8, 16]],
                                  ["≤"] * 3, [60, 30, 32]), pivot_rule="professor")
        steps = session.original.steps
        self.assertEqual([(s.row, s.column) for s in steps], [(1, 1), (2, 0), (None, None)])
        np.testing.assert_allclose(session.original.point(2), [10/3, 1/3])
        np.testing.assert_allclose(session.original_prices, [0, 0, 2.5], atol=1e-12)
        change = session.apply([66, 30, 25.6])
        self.assertEqual(change.method, "Preço-sombra")
        app = render(session)
        self.assertFalse(app.exception)
        for i, price in enumerate(["0", "0", "2,5"], 1):
            self.assertIn(f"PS(R{i}) = {price}", displayed_text(app, 2))
        self.assertIn("ZNOVO = 64", displayed_text(app, 3))

    def test_interface_uses_professor_criterion_by_default(self):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.number_input(key="c0").set_value(20.)
        app.number_input(key="c1").set_value(40.)
        next(button for button in app.button if button.label == "Resolver PPL").click().run()
        self.assertFalse(app.exception)
        session = app.session_state["session"]
        self.assertEqual(session.pivot_rule, "professor")
        self.assertEqual(session.original.steps[0].column, 1)

    def test_failed_condition_and_dual_result_are_displayed(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        session.apply([170, 80])
        app = render(session)
        self.assertFalse(app.exception)
        text = displayed_text(app, 3)
        self.assertIn("60 − (70) + 2(0) = -10  (NÃO VIÁVEL)", text)
        self.assertIn("ZNOVO = 3200", text)
        self.assertNotIn("ZNOVO = 2600 + 10(70)", text)


    def test_mixed_resource_operations_and_restore_in_interface(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        app = render(session)

        def widget(kind, label):
            return next(item for item in getattr(app, kind) if item.label == label)

        def select_operation(i, operation, unit, value):
            widget("selectbox", f"Operação de R{i}").set_value(operation).run()
            widget("selectbox", f"Unidade de R{i}").set_value(unit).run()
            widget("number_input", f"Valor de R{i}").set_value(value)

        def submit():
            widget("button", "Calcular novo Z").click().run()
            self.assertFalse(app.exception)

        app.multiselect[0].set_value([0, 1]).run()
        select_operation(1, "Aumentar", "%", 20.)
        select_operation(2, "Diminuir", "%", 10.)
        with patch("metodo_simplex._run", side_effect=AssertionError("Pivoteamento inesperado")):
            submit()
            np.testing.assert_allclose(session.rhs, [120, 72])
            self.assertIn("ZNOVO = 2640", displayed_text(app, 3))
            app.multiselect[0].set_value([0]).run()
            select_operation(1, "Diminuir", "unidades", 10.)
            submit()
        np.testing.assert_allclose(session.rhs, [110, 72])
        self.assertIn("ZNOVO = 2540", displayed_text(app, 3))
        app.multiselect[0].set_value([0]).run()
        select_operation(1, "Aumentar", "vezes", 2.)
        submit()
        np.testing.assert_allclose(session.rhs, [220, 72])
        self.assertEqual(session.history[-1].method, "Simplex Dual")
        self.assertIn("NÃO VIÁVEL", displayed_text(app, 3))
        self.assertIn("ZNOVO = 2880", displayed_text(app, 3))
        history = app.tabs[4].dataframe[0].value
        np.testing.assert_allclose(history["Variação (%)"].iloc[:2], [20, -10])
        np.testing.assert_allclose(history["Δ"].iloc[:2], [20, -8])
        with patch("metodo_simplex._run", side_effect=AssertionError("Pivoteamento inesperado")):
            widget("button", "Restaurar problema original").click().run()
        self.assertFalse(app.exception)
        self.assertIs(session.current, session.original)
        np.testing.assert_allclose(session.rhs, [100, 80])
        self.assertEqual(len(session.history), 4)
        self.assertEqual(session.history[-1].method, "Restauração")
        self.assertIn("ZNOVO = 2600", displayed_text(app, 3))


    def test_bulk_actions_saved_state_and_pdf_download(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        app = render(session)

        def click(label):
            next(button for button in app.button if button.label == label).click().run()
            self.assertFalse(app.exception)

        with patch("metodo_simplex._run", side_effect=AssertionError("Pivoteamento inesperado")):
            click("Diminuir todos os recursos em X%")
            saved = session.current
            np.testing.assert_allclose(session.rhs, [90, 72])
            self.assertIn("ZNOVO = 2340", displayed_text(app, 3))
            click("Aumentar todos os recursos em X%")
            np.testing.assert_allclose(session.rhs, [99, 79.2])
            click("Restaurar estado selecionado")
        self.assertIs(session.current, saved)
        self.assertEqual(len(session.history), 3)
        np.testing.assert_allclose(session.rhs, [90, 72])
        self.assertEqual(session.history[0].operations[0], (0, "Diminuir", "%", 10))
        click("Preparar PDF")
        self.assertEqual(len(app.sidebar.get("download_button")), 1)
        for section in REPORTS:
            with self.subTest(section=section):
                data = report_pdf(session, section, lambda value: f"{value:g}")
                self.assertTrue(data.startswith(b"%PDF-"))
                self.assertTrue(data.rstrip().endswith(b"%%EOF"))

    def test_manual_profit_shows_each_product(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        session.apply([120, 72], operations=((0, "Aumentar", "%", 20), (1, "Diminuir", "%", 10)))
        app = render(session)
        self.assertFalse(app.exception)
        calculations = displayed_text(app, 3)
        self.assertIn("Δ1 = (100) × 20 / 100 = +20", calculations)
        self.assertIn("Δ2 = −(80) × 10 / 100 = -8", calculations)
        self.assertIn("ZNOVO = 2600 + 200 − 160", calculations)
        self.assertIn("ZNOVO = 2640", calculations)



    def test_form_submits_latest_amount_instead_of_previous_preview(self):
        session = Session(Problem([40, 30], [[2, 1], [1, 1]], ["≤"] * 2, [100, 80]))
        app = render(session)
        app.multiselect[0].set_value([0, 1]).run()

        def widget(kind, label):
            return next(item for item in getattr(app, kind) if item.label == label)

        widget("selectbox", "Operação de R1").set_value("Aumentar")
        widget("number_input", "Valor de R1").set_value(10.)
        widget("selectbox", "Operação de R2").set_value("Diminuir")
        widget("number_input", "Valor de R2").set_value(10.)
        widget("button", "Atualizar prévia").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Novo LD1 = 110", displayed_text(app, 3))
        self.assertIn("Novo LD2 = 72", displayed_text(app, 3))
        self.assertEqual(len(session.history), 0)
        widget("number_input", "Valor de R1").set_value(20.)
        with patch("metodo_simplex._run", side_effect=AssertionError("Pivoteamento inesperado")):
            widget("button", "Calcular novo Z").click().run()
        self.assertFalse(app.exception)
        change = session.history[-1]
        np.testing.assert_allclose(change.new_rhs, [120, 72])
        np.testing.assert_allclose(change.trial, [48, 24])
        self.assertEqual(change.violations, ())
        self.assertEqual(change.new_z, 2640)
        self.assertEqual(change.method, "Preço-sombra")
        self.assertNotIn("Novo LD1 = 110", displayed_text(app, 3))

    def test_formatting_never_mutates_the_tableau(self):
        session = Session(Problem([1, 0], [[3, 0]], ["≤"], [1]))
        app = render(session)
        self.assertFalse(app.exception)
        digits = next(item for item in app.slider if item.label == "Casas decimais")
        digits.set_value(0).run()
        self.assertEqual(app.tabs[1].metric[0].value, "0")
        next(item for item in app.slider if item.label == "Casas decimais").set_value(8).run()
        self.assertEqual(app.tabs[1].metric[0].value, "0,33333333")
        self.assertEqual(session.original.point(2)[0], 1 / 3)


class ResourceChangeTests(unittest.TestCase):
    def test_number_format(self):
        for value, digits, expected in (
            (-0., 4, "0"), (-1e-8, 8, "0"), (20., 4, "20"),
            (10.5, 4, "10,5"), (1/3, 4, "0,3333"),
            (-.1, 0, "0"), (100., 0, "100"), (np.inf, 4, "∞"),
            (np.nan, 4, "—"), ("VB", 4, "VB"),
        ):
            with self.subTest(value=value, digits=digits):
                self.assertEqual(format_number(value, digits), expected)

    def test_resource_conversions(self):
        for operation, unit, value, expected in (
            ("Aumentar", "%", 20, 120),
            ("Diminuir", "%", 10, 90),
            ("Aumentar", "unidades", 50, 150),
            ("Diminuir", "unidades", 20, 80),
            ("Aumentar", "vezes", 2, 200),
            ("Diminuir", "vezes", .5, 50),
            ("Definir", "LD final", -10, -10),
            ("Dobrar", "vezes", 2, 200),
            ("Multiplicar", "vezes", .2, 20),
            ("Restaurar LD original", "LD final", 80, 80),
        ):
            with self.subTest(operation=operation, unit=unit):
                self.assertEqual(resource_rhs(100, operation, unit, value), expected)
        self.assertEqual(resource_rhs(0, "Aumentar", "%", 20), 0)
        self.assertIsNone(relative_percentage(0, 20))
        with self.assertRaises(ValueError):
            resource_rhs(100, "Diminuir", "vezes", 2)
