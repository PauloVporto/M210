"""Formatação dos quadros e das expressões usadas na resolução manual."""

from math import isinf, isnan
from numbers import Real

from contas_recursos import operation_calculation, operation_description
from quadro_simplex import TOL


DISPLAY_TOL = 1e-8


def format_number(value, digits=4):
    """Formata células e números em português, sem modificar o valor de cálculo."""
    if value is None:
        return "—"
    if not isinstance(value, Real):
        return str(value)
    if isnan(value):
        return "—"
    if isinf(value):
        return "−∞" if value < 0 else "∞"
    if abs(value) <= DISPLAY_TOL:
        return "0"
    rendered = f"{value:.{digits}f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered == "-0" else rendered.replace(".", ",")


def manual_columns(tableau):
    """Nomeia as colunas na convenção VB, variáveis e LD da apostila."""
    return ["VB", *(name.upper() for name in tableau.names), "LD"]


def manual_rows(tableau, objective_sign=1):
    """Emite a linha Z antes das linhas básicas, sem arredondar coeficientes."""
    labels = [tableau.names[j].upper() for j in tableau.basis]
    yield ["Z", *(-objective_sign * tableau.reduced), objective_sign * tableau.z]
    for label, row, rhs in zip(labels, tableau.coefficients, tableau.rhs):
        yield [label, *row, rhs]


def delta_expression(constant, coefficients, fmt, delta=None, *, objective=False):
    """Monta a forma simbólica ou a mesma expressão com os Δ substituídos."""
    parts = [fmt(constant)]
    for i, coefficient in enumerate(coefficients):
        if coefficient == 0:
            continue
        sign = "−" if coefficient < 0 else "+"
        magnitude = abs(coefficient)
        factor = fmt(magnitude) if magnitude != 1 or objective else ""
        variable = f"Δ{i + 1}" if delta is None else f"({fmt(delta[i])})"
        parts.append(f"{sign} {factor}{variable}")
    return " ".join(parts)


def objective_scale(step, direction):
    return 1 if step.phase in ("Fase 1", "Transição") else direction


def pivot_lines(step, direction, fmt):
    t = step.tableau
    row, col = step.row, step.column
    if row is None:
        return
    pivot = t.coefficients[row, col]
    yield (f"Linha pivô L{row + 1} ({t.names[t.basis[row]]}) · "
           f"Coluna pivô {t.names[col]} · Elemento {fmt(pivot)}")
    if step.phase == "Simplex Dual":
        yield "Linha com LD negativo pela regra de Bland. Coluna pela menor razão dual."
        for j, ratio in step.ratios:
            yield (f"{t.names[j]}: {fmt(t.reduced[j])} / "
                   f"({fmt(t.coefficients[row, j])}) = {fmt(ratio)}")
    elif step.ratios:
        if step.pivot_rule == "professor":
            yield "Coluna pelo coeficiente mais negativo da linha Z interna (Zj − Cj)."
            yield "Em empate nas razões, escolhe-se a primeira linha do quadro."
        else:
            yield "Coluna pela regra de Bland."
        yield "Linha pela menor razão LD / coeficiente positivo."
        for i, ratio in step.ratios:
            yield (f"{t.names[t.basis[i]]}: {fmt(t.rhs[i])} / "
                   f"{fmt(t.coefficients[i, col])} = {fmt(ratio)}")
    rows = [list(item[1:]) for item in manual_rows(t, objective_scale(step, direction))]
    pivot_values = [value / pivot for value in rows[row + 1]]

    def vector(values):
        return "[" + "; ".join(fmt(value) for value in values) + "]"

    yield f"L{row + 1}′ = L{row + 1} / ({fmt(pivot)})"
    yield f"{vector(rows[row + 1])} / ({fmt(pivot)}) = {vector(pivot_values)}"
    for i, values in enumerate(rows[1:]):
        if i != row:
            coefficient = t.coefficients[i, col]
            yield f"L{i + 1}′ = L{i + 1} − ({fmt(coefficient)}) × L{row + 1}′"
            updated = [value - coefficient * part for value, part in zip(values, pivot_values)]
            yield f"{vector(values)} − ({fmt(coefficient)}) × {vector(pivot_values)} = {vector(updated)}"
    coefficient = rows[0][col]
    yield f"LZ′ = LZ − ({fmt(coefficient)}) × L{row + 1}′"
    updated = [value - coefficient * part for value, part in zip(rows[0], pivot_values)]
    yield f"{vector(rows[0])} − ({fmt(coefficient)}) × {vector(pivot_values)} = {vector(updated)}"


def conditions(result, fmt, change=None):
    t = result.tableau
    for i, (name, rhs, coefficients) in enumerate(zip(result.basic_names, t.rhs, t.inverse)):
        relation = "= 0" if i in result.fixed else "≥ 0"
        yield f"({name})  {delta_expression(rhs, coefficients, fmt)} {relation}", None
        if change is not None:
            ok = name not in change.violations
            value = 0 if abs(change.trial[i]) <= TOL else change.trial[i]
            computed = delta_expression(rhs, coefficients, fmt, change.delta)
            result_text = f"{computed} = {fmt(value)}" + (f" {relation}" if ok else "")
            note = " · fora da tolerância antes do arredondamento" if not ok and fmt(value) == "0" else ""
            yield f"        {result_text}  ({'OK' if ok else 'NÃO VIÁVEL'}){note}", ok


def objective_lines(change, fmt):
    if change.method == "Preço-sombra":
        terms = " + ".join(f"PS{i}×Δ{i}" for i in range(1, len(change.shadow) + 1))
        yield f"ZNOVO = ZANTERIOR + {terms}"
        yield f"ZNOVO = {delta_expression(change.old_z, change.shadow, fmt, objective=True)}"
        yield f"ZNOVO = {delta_expression(change.old_z, change.shadow, fmt, change.delta, objective=True)}"
        products = [f"{'−' if y * d < 0 else '+'} {fmt(abs(y * d))}"
                    for y, d in zip(change.shadow, change.delta) if y != 0]
        if products:
            yield f"ZNOVO = {fmt(change.old_z)} {' '.join(products)}"
    if change.new_z is not None:
        yield f"ZNOVO = {fmt(change.new_z)}"


def objective_expression(coefficients, fmt):
    return " ".join(f"{'−' if c < 0 else '+'} {fmt(abs(c))}x{i + 1}"
                    for i, c in enumerate(coefficients) if c).removeprefix("+ ") or "0"


def proposal_lines(change, fmt):
    operations = {i: (operation, unit, value) for i, operation, unit, value in change.operations}
    for i, (old, new, delta) in enumerate(zip(change.old_rhs, change.new_rhs, change.delta)):
        n = i + 1
        yield f"R{n} · LD anterior = {fmt(old)}"
        if i in operations:
            operation, quantity = operation_description(*operations[i], fmt)
            yield f"{operation} · {quantity}"
            calculation = operation_calculation(old, *operations[i], fmt)
        else:
            calculation = f"{fmt(new)} − ({fmt(old)})"
        signed = ("+" if delta > 0 else "") + fmt(delta)
        yield f"Δ{n} = {calculation} = {signed}"
        yield f"Novo LD{n} = {fmt(new)}"
