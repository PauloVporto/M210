"""Relatórios PDF gerados a partir dos quadros e das tentativas armazenadas."""

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from exibicao import (conditions, delta_expression, manual_columns, manual_rows, objective_expression, objective_lines,
                          objective_scale, pivot_lines, proposal_lines)
from analise_sensibilidade import prices, ranges


REPORTS = {
    "completo": "Resolução completa",
    "quadros": "Quadros Simplex",
    "sensibilidade": "Análise de sensibilidade",
    "historico": "Histórico das alterações",
}


def report_pdf(session, section, fmt, *, sensitivity_reference="Atual"):
    """Gera um relatório em memória usando os quadros e as fórmulas da interface."""
    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=landscape(A4), title=REPORTS[section])
    style = getSampleStyleSheet()
    content = []
    problem = session.problem

    def text(value, kind="BodyText"):
        content.append(Paragraph(escape(value), style[kind]))

    def tableau(t, scale):
        rows = list(manual_rows(t, scale))
        columns = manual_columns(t)
        # Quadros com muitas restrições são divididos em grupos de colunas.
        for start in range(0, len(t.names), 10):
            stop = min(start + 10, len(t.names))
            data = [[columns[0], *columns[start + 1:stop + 1], columns[-1]]]
            for row in rows:
                data.append([row[0], *(fmt(v) for v in row[start + 1:stop + 1]), fmt(row[-1])])
            table = Table(data, repeatRows=1, colWidths=[document.width / len(data[0])] * len(data[0]))
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDEAE8")),
                ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#AABDBB")),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]))
            content.extend([table, Spacer(1, 6)])

    def steps(result):
        for i, step in enumerate(result.steps, 1):
            title = f"{i}º Quadro · {step.phase}"
            if i == len(result.steps) and result.status == "ótimo":
                title += " · Quadro Ótimo"
            text(title, "Heading3")
            tableau(step.tableau, objective_scale(step, problem.direction))
            if step.note:
                text(step.note)
            for line in pivot_lines(step, problem.direction, fmt):
                text(line)

    def point(result):
        for i, value in enumerate(result.point(problem.c.size), 1):
            text(f"x{i} = {fmt(value)}")

    def shadow(result):
        for i, value in enumerate(prices(problem, result), 1):
            text(f"PS(R{i}) = {fmt(value)}")

    text(REPORTS[section], "Title")
    text("Modelo matemático", "Heading2")
    sense = "Maximizar" if problem.sense == "max" else "Minimizar"
    text(f"{sense} Z = {objective_expression(problem.c, fmt)}")
    for i, (left, sign, right) in enumerate(zip(problem.a, problem.signs, problem.b), 1):
        text(f"R{i} · {objective_expression(left, fmt)} {sign} {fmt(right)}")
    text("Variáveis de decisão ≥ 0")

    if section in ("completo", "quadros"):
        text("Resolução Simplex", "Heading2")
        steps(session.original)
        text(f"Resultado: {session.original.status}")
        if session.original.status == "ótimo":
            text(f"Z ótimo = {fmt(problem.direction * session.original.tableau.z)}")
            point(session.original)
        if section == "quadros":
            for i, change in enumerate(session.history, 1):
                if change.method == "Simplex Dual":
                    text(f"Alteração {i}", "Heading2")
                    steps(change.result)

    if section in ("completo", "sensibilidade"):
        original = section == "sensibilidade" and sensitivity_reference == "Original"
        result, rhs = (session.original, problem.b) if original else (session.current, session.rhs)
        scope = "original" if original else "vigente"
        text(f"Análise de sensibilidade · quadro {scope}", "Heading2")
        if result.status == "ótimo":
            tableau(result.tableau, problem.direction)
            text("Preço-Sombra", "Heading3")
            shadow(result)
            for i, (low, high) in enumerate(ranges(result, rhs), 1):
                text(f"Faixa de R{i} · {fmt(low)} a {fmt(high)}")
            text("CONDIÇÕES DE VIABILIDADE", "Heading3")
            for line, _ in conditions(result, fmt):
                text(line)
            z = problem.direction * result.tableau.z
            formula = delta_expression(z, prices(problem, result), fmt, objective=True)
            text(f"ZNOVO = {formula}")
        else:
            text("A análise de LD requer uma solução ótima de referência.")

    if section in ("completo", "historico"):
        text("Histórico das alterações", "Heading2")
        if not session.history:
            text("Nenhuma alteração registrada.")
        for i, change in enumerate(session.history, 1):
            text(f"Alteração {i} · {change.method}", "Heading2")
            text("ALTERAÇÃO PROPOSTA", "Heading3")
            for line in proposal_lines(change, fmt):
                text(line)
            text("CONDIÇÕES DE VIABILIDADE", "Heading3")
            for line, _ in conditions(change.previous, fmt, change):
                text(line)
            base = "inviável para a proposta" if change.violations else "viável para a proposta"
            text(f"Base anterior {base}. Resultado do problema modificado: {change.result.status}.")
            if change.method == "Simplex Dual":
                steps(change.result)
            text(f"Z anterior = {fmt(change.old_z)}")
            for line in objective_lines(change, fmt):
                text(line)
            if change.new_z is not None:
                text("Valores Ótimos", "Heading3")
                for name, value in zip(change.result.basic_names, change.result.tableau.rhs):
                    text(f"{name} = {fmt(value)}")
                point(change.result)
                if change.method != "Preço-sombra":
                    shadow(change.result)

    document.build(content)
    return output.getvalue()
