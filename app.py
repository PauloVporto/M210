"""Interface didática do Simplex Tableau."""

import pandas as pd
import streamlit as st

from presentation import (conditions, delta_expression, format_number, manual_columns, manual_rows, objective_expression,
                          objective_lines, objective_scale, pivot_lines, proposal_lines)
from reports import REPORTS, report_pdf
from resource_changes import (operation_calculation, operation_description, relative_percentage,
                              resource_rhs)
from sensitivity import prices, ranges, slacks
from simplex import Problem
from state_manager import Session
from tableau import TOL

st.set_page_config(page_title="Simplex Tableau", layout="wide")
st.title("Simplex Tableau")
settings = st.sidebar
digits = settings.slider("Casas decimais", 0, 8, 4)
show_costs = settings.checkbox("Mostrar custos e linha Zj")
convention = (settings.radio("Custos reduzidos", ["Cj − Zj", "Zj − Cj"])
              if show_costs else "Cj − Zj")


def fmt(value):
    """Aplica à tela e ao PDF a precisão escolhida na barra lateral."""
    return format_number(value, digits)


def grid(rows, columns=None, index=None):
    """Exibe uma tabela com a formatação de apresentação e dados originais."""
    frame = pd.DataFrame(rows, columns=columns, index=index)
    st.dataframe(frame.style.format(fmt))


def point(problem, result, z):
    st.metric("Z ótimo", fmt(z))
    grid([result.point(problem.c.size)], [f"x{i + 1}" for i in range(problem.c.size)])


def show_tableau(tableau, objective_sign=1):
    grid(manual_rows(tableau, objective_sign), manual_columns(tableau))


def show_steps(steps, direction, optimal=False, *, context="original"):
    for i, step in enumerate(steps):
        t = step.tableau
        last = i == len(steps) - 1
        title = f"{i + 1}º Quadro"
        if last and optimal:
            title += " · Quadro Ótimo"
        objective_sign = objective_scale(step, direction)
        with st.expander(title, expanded=i == 0 or last):
            st.caption("Quadro anterior com LD atualizado · Simplex Dual"
                       if i == 0 and step.phase == "Simplex Dual" else step.phase)
            show_tableau(t, objective_sign)
            if show_costs:
                reduced = t.reduced if convention == "Cj − Zj" else -t.reduced
                grid([t.costs, t.zj, reduced], t.names, ["Cj", "Zj", convention])
            if step.note:
                st.caption(step.note)
            previous = steps[i - 1] if i else None
            if previous is not None and previous.row is not None and st.checkbox(
                    "Como este quadro foi calculado?", key=f"pivot_{context}_{i}"):
                for line in pivot_lines(previous, direction, fmt):
                    st.text(line)


def show_conditions(result, change=None):
    for line, ok in conditions(result, fmt, change):
        st.text(line)
        if ok is not None:
            st.markdown(":green[✓ OK]" if ok else ":red[✗ NÃO VIÁVEL]")


def show_change(problem, change, *, context):
    labels = [f"R{i + 1}" for i in range(len(change.delta))]
    st.subheader("ALTERAÇÃO PROPOSTA")
    for line in proposal_lines(change, fmt):
        st.text(line)
    st.subheader("CONDIÇÕES DE VIABILIDADE")
    show_conditions(change.previous, change)
    objective = "CÁLCULO DO NOVO LUCRO" if problem.sense == "max" else "CÁLCULO DO NOVO VALOR ÓTIMO"
    restored = change.method == "Restauração"
    if restored:
        st.success("Quadro ótimo salvo restaurado.")
        if change.violations:
            st.caption("Condições da base anterior que falharam: " + ", ".join(change.violations))
    elif change.violations:
        st.warning("A base ótima anterior perdeu a viabilidade nas condições de "
                   f"{', '.join(change.violations)}. Os preços-sombra anteriores não "
                   "podem ser usados diretamente. O Simplex Dual parte do quadro atualizado.")
        show_steps(change.result.steps, problem.direction, change.new_z is not None,
                   context=context)
    else:
        st.success("A alteração é viável para a base ótima atual.")
    if change.new_z is not None:
        st.subheader(objective)
        for line in objective_lines(change, fmt):
            st.text(line)
    if change.new_z is None:
        st.error(f"{show_status(change.result.status, modified=True)} "
                 "Os LDs da última solução ótima seguem ativos.")
    else:
        st.subheader("Valores Ótimos")
        t = change.result.tableau
        for name, value in zip(change.result.basic_names, t.rhs):
            st.text(f"{name} = {fmt(0 if abs(value) <= TOL else value)}")
        point(problem, change.result, change.new_z)
        if restored or change.violations:
            st.subheader("Preço-Sombra")
            for i, value in enumerate(prices(problem, change.result), 1):
                st.text(f"PS(R{i}) = {fmt(value)}")
    with st.expander("Detalhes matemáticos"):
        st.write(f"Z inicial **{fmt(problem.direction * session.original.tableau.z)}** · "
                 f"Z anterior **{fmt(change.old_z)}** · Z novo **{fmt(change.new_z)}**")
        grid(zip(change.old_rhs, change.new_rhs, change.delta, change.shadow,
                 *change.previous_ranges.T),
             ["LD anterior", "LD proposto", "ΔLD", "Preço-sombra anterior", "Mín.", "Máx."],
             labels)
        st.latex(r"x_{B,novo}=x_{B,anterior}+B^{-1}\Delta b")
        st.markdown("**B**")
        grid(change.previous.base_matrix, change.previous.basic_names, labels)
        st.markdown("**B⁻¹**")
        grid(change.previous.tableau.inverse, labels, change.previous.basic_names)


def resource_proposal(session, operations):
    """Converte o formulário confirmado em um vetor de LDs, sem alterar a sessão."""
    proposal = session.rhs.copy()
    for i, operation, unit, value in operations:
        proposal[i] = resource_rhs(float(session.rhs[i]), operation, unit, value)
    return proposal


def show_status(status, *, modified=False):
    """Explica por que o problema não tem um ótimo disponível para exibição."""
    subject = "Problema modificado" if modified else "Problema original"
    if status == "inviável":
        return f"{subject} inviável. As restrições não admitem uma solução em conjunto."
    if status == "ilimitado":
        return f"{subject} ilimitado. A função objetivo pode melhorar sem limite."
    return "Falha numérica. Não foi possível confirmar uma solução ótima com estes coeficientes."


entrada, resolucao, sensibilidade, alteracoes, historico = st.tabs(
    ["Entrada do PPL", "Resolução Simplex", "Análise de sensibilidade",
     "Alterações do LD", "Histórico"]
)
with entrada:
    col1, col2, col3 = st.columns(3)
    sense = col1.selectbox("Sentido", ["Maximizar", "Minimizar"])
    n = col2.selectbox("Variáveis de decisão", [2, 3, 4])
    m = col3.number_input("Restrições", min_value=1, value=3, step=1)
    with st.form("ppl"):
        rule = st.radio("Critério para a coluna pivô", ["Modelo do professor", "Regra de Bland"],
                        horizontal=True)
        st.subheader("Função objetivo")
        c = [col.number_input(f"x{j + 1}", value=float([3, 2, 0, 0][j]), key=f"c{j}")
             for j, col in enumerate(st.columns(n))]
        st.subheader("Restrições")
        a, signs, b = [], [], []
        for i in range(m):
            st.write(f"**R{i + 1}**")
            cols = st.columns(n + 2)
            defaults = [(1, 1), (1, 0), (0, 1)]
            a.append([cols[j].number_input(
                f"x{j + 1}", value=float(defaults[i][j]) if i < 3 and j < 2 else 0.,
                key=f"a{i}_{j}") for j in range(n)])
            signs.append(cols[n].selectbox("Sinal", ["≤", "≥", "="], key=f"s{i}"))
            b.append(cols[n + 1].number_input(
                "LD", value=float([4, 2, 3][i]) if i < 3 else 0., key=f"b{i}"))
        if st.form_submit_button("Resolver PPL", type="primary"):
            try:
                problem = Problem(c, a, signs, b, "max" if sense == "Maximizar" else "min")
                st.session_state["session"] = Session(
                    problem, pivot_rule="professor" if rule == "Modelo do professor" else "bland")
                st.session_state["model_version"] = st.session_state.get("model_version", 0) + 1
                st.success("Resolução disponível na aba Resolução Simplex.")
            except ValueError as error:
                st.error(str(error))

session = st.session_state.get("session")
if session is None:
    for tab in (resolucao, sensibilidade, alteracoes, historico):
        with tab:
            st.info("Preencha e resolva o PPL na aba Entrada do PPL.")
    st.stop()

problem = session.problem
with settings.expander("Exportar resultados"):
    report = st.selectbox("Relatório", list(REPORTS), format_func=REPORTS.get)
    reference = st.session_state.get("sensitivity_reference", "Atual") if report == "sensibilidade" else "Atual"
    export_key = (st.session_state["model_version"], len(session.history), report, digits, reference)
    if st.button("Preparar PDF"):
        document = report_pdf(session, report, fmt, sensitivity_reference=reference)
        st.session_state["prepared_pdf"] = (export_key, document)
    prepared = st.session_state.get("prepared_pdf")
    if prepared is not None and prepared[0] == export_key:
        st.download_button(f"Baixar {REPORTS[report]} em PDF", prepared[1],
                           file_name=f"simplex-{report}.pdf", mime="application/pdf")

with resolucao:
    st.subheader("Problema original")
    st.text(f"{'Maximizar' if problem.sense == 'max' else 'Minimizar'} Z = {objective_expression(problem.c, fmt)}")
    for i, (a_i, sign, b_i) in enumerate(zip(problem.a, problem.signs, problem.b)):
        st.text(f"R{i + 1}   {objective_expression(a_i, fmt)} {sign} {fmt(b_i)}")
    st.text("Todas as variáveis de decisão ≥ 0")
    if any(problem.b < 0):
        st.caption("As linhas com LD negativo foram multiplicadas por −1 no quadro inicial.")
    if problem.sense == "min":
        st.caption("Na Fase 2, a linha Z está na escala da função objetivo original.")
    optimal = session.original.status == "ótimo"
    show_steps(session.original.steps, problem.direction, optimal)
    if optimal:
        st.subheader("Valores Ótimos")
        point(problem, session.original, problem.direction * session.original.tableau.z)
    else:
        st.error(show_status(session.original.status))

if session.original.status != "ótimo":
    for tab in (sensibilidade, alteracoes, historico):
        with tab:
            st.info("A análise de LD requer uma solução ótima de referência.")
    st.stop()

with sensibilidade:
    which = st.radio("Solução de referência", ["Atual", "Original"], horizontal=True,
                     key="sensitivity_reference")
    result, rhs = (session.current, session.rhs) if which == "Atual" else (
        session.original, problem.b)
    labels = [f"R{i + 1}" for i in range(len(rhs))]
    st.subheader("Quadro Ótimo")
    t = result.tableau
    show_tableau(t, problem.direction)
    st.subheader("Preço-Sombra")
    shadow = prices(problem, result)
    for i, value in enumerate(shadow):
        suffix = str(i + 1)
        column = next((name for name in (f"f{suffix}", f"e{suffix}") if name in t.names), None)
        if column:
            j = t.names.index(column)
            original_sign = result.matrix[i, j]
            z_coefficient = -problem.direction * t.reduced[j]
            source = f"Linha Z, coluna {column.upper()}"
            if original_sign < 0:
                source += f" com sinal invertido (−({fmt(z_coefficient)}))"
        else:
            source = "Linha Z transformada · CbᵀB⁻¹"
        st.text(f"PS(R{suffix}) = {fmt(value)}  ·  {source}")
    st.subheader("Condições de Viabilidade")
    st.caption("Cada Δ representa a variação em relação ao LD do quadro de referência.")
    show_conditions(result)
    st.text(f"ZNOVO = {delta_expression(problem.direction * t.z, shadow, fmt, objective=True)}")
    with st.expander("Faixas de Viabilidade e Folgas"):
        lower, upper = ranges(result, rhs).T
        st.caption("Cada faixa mantém os demais LDs fixos.")
        grid(zip(problem.b, rhs, lower, upper, rhs - lower, upper - rhs,
                 shadow, slacks(problem, result, rhs)),
             ["LD original", "LD de referência", "LD mínimo", "LD máximo", "Diminuição",
              "Aumento", "Preço-sombra", "Folga / excesso"], labels)
    with st.expander("Detalhes matemáticos"):
        st.latex(r"y^\top=C_B^\top B^{-1},\qquad x_B+B^{-1}_{:,i}\Delta b_i\geq 0")
        st.write(f"Básicas · {', '.join(result.basic_names)}")
        st.write(f"Não básicas · {', '.join(result.nonbasic_names) or 'nenhuma'}")
        if result.fixed:
            st.caption("Artificiais de linhas dependentes ficam fixas em zero.")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**B**")
            grid(result.base_matrix, result.basic_names, labels)
        with cols[1]:
            st.markdown("**B⁻¹**")
            grid(t.inverse, labels, result.basic_names)
        grid([problem.direction * t.cb, t.rhs], result.basic_names, ["Cb", "Valor básico"])

with alteracoes:
    st.subheader("Propor novos LDs")
    model_version = st.session_state["model_version"]
    version = f"{model_version}_{len(session.history)}"
    selected = st.multiselect(
        "Restrições", list(range(len(session.rhs))), format_func=lambda i: f"R{i + 1}",
        key=f"selected_{model_version}",
    )
    st.caption("Selecione os recursos antes de editar. Os campos são enviados juntos ao "
               "clicar em Atualizar prévia ou Calcular novo Z. A prévia mostra apenas o "
               "último envio e não aplica a proposta.")
    operations = []
    with st.form(f"resource_{model_version}"):
        for i in selected:
            label = f"R{i + 1}"
            current = float(session.rhs[i])
            st.write(f"**{label}** · LD atual {fmt(current)} · LD original {fmt(problem.b[i])}")
            cols = st.columns(3)
            operation = cols[0].selectbox(
                f"Operação de {label}", ["Definir", "Aumentar", "Diminuir", "Dobrar",
                                        "Multiplicar", "Restaurar LD original"],
                format_func=lambda name: "Definir novo LD" if name == "Definir" else name,
                key=f"operation_{model_version}_{i}",
            )
            unit = cols[1].selectbox(f"Unidade de {label}", ["%", "unidades", "vezes"],
                                     key=f"unit_{model_version}_{i}",
                                     help="Usada em Aumentar e Diminuir.")
            value_key = f"value_{model_version}_{i}"
            st.session_state.setdefault(value_key, current)
            value = cols[2].number_input(
                f"Valor de {label}", format="%.10g", key=value_key,
                help="LD final em Definir, fator em Multiplicar. Dobrar e Restaurar ignoram o valor.",
            )
            if operation == "Restaurar LD original":
                value = float(problem.b[i])
            elif operation == "Dobrar":
                value = 2.
            if operation in ("Dobrar", "Multiplicar"):
                unit = "vezes"
            elif operation in ("Definir", "Restaurar LD original"):
                unit = "LD final"
            operations.append((i, operation, unit, value))
        left, right = st.columns(2)
        preview = left.form_submit_button("Atualizar prévia", disabled=not selected)
        calculate = right.form_submit_button("Calcular novo Z", disabled=not selected, type="primary")

    preview_key = (version, tuple(selected))
    if preview or calculate:
        st.session_state.pop("resource_preview", None)
        try:
            proposal = resource_proposal(session, operations)
            if calculate:
                session.apply(proposal, operations=operations)
                st.rerun()
            st.session_state["resource_preview"] = (preview_key, tuple(operations), proposal)
        except ValueError as error:
            st.error(f"Dados da alteração inválidos. {error}")
        except ArithmeticError as error:
            st.error(f"Falha numérica. {error}")
    saved = st.session_state.get("resource_preview")
    if saved is not None and saved[0] == preview_key:
        st.subheader("Última prévia confirmada · proposta não aplicada")
        for i, operation, unit, value in saved[1]:
            current, new = session.rhs[i], saved[2][i]
            expression = operation_calculation(current, operation, unit, value, fmt)
            st.text(f"Δ{i + 1} = {expression} = {fmt(new - current)}")
            st.text(f"Novo LD{i + 1} = {fmt(new)}")

    if st.button("Restaurar problema original", disabled=session.current is session.original):
        session.restore()
        st.rerun()
    with st.expander("Alterar todos os recursos"):
        with st.form("bulk_resources"):
            percent = st.number_input("Percentual para todos os recursos", min_value=0., value=10.)
            left, right = st.columns(2)
            increase = left.form_submit_button("Aumentar todos os recursos em X%")
            decrease = right.form_submit_button("Diminuir todos os recursos em X%")
        if increase or decrease:
            operation = "Aumentar" if increase else "Diminuir"
            batch = [(i, operation, "%", percent) for i in range(len(session.rhs))]
            try:
                session.apply(resource_proposal(session, batch), operations=batch)
                st.rerun()
            except ValueError as error:
                st.error(f"Dados da alteração inválidos. {error}")
            except ArithmeticError as error:
                st.error(f"Falha numérica. {error}")
    if session.history:
        show_change(problem, session.history[-1], context=f"latest_{version}")

with historico:
    if not session.history:
        st.info("As alterações de LD aparecerão aqui.")
    else:
        rows = []
        for number, change in enumerate(session.history, start=1):
            viable = "NÃO VIÁVEL" if change.violations else "OK"
            restored = change.method == "Restauração"
            details = {i: operation_description(op, unit, value, fmt)
                       for i, op, unit, value in change.operations}
            indices = [j for j, delta in enumerate(change.delta) if delta or j in details]
            for i in indices or [None]:
                previous = change.old_rhs[i] if i is not None else None
                updated = change.new_rhs[i] if i is not None else None
                delta = change.delta[i] if i is not None else None
                percentage = relative_percentage(previous, delta) if i is not None else None
                shadow = (change.shadow[i] if i is not None and change.method == "Preço-sombra"
                          else None)
                kind, quantity = details.get(i, (change.method if restored else "LD direto", "—"))
                rows.append((number, f"R{i + 1}" if i is not None else "Sem mudança",
                             kind, quantity, percentage, delta, previous, updated, viable, shadow,
                             change.old_z, change.new_z,
                             "Restaurada" if restored else (
                                 "Mantida" if not change.violations else (
                                     "Atualizada" if change.new_z is not None else change.result.status)),
                             change.method))
        grid(rows, ["Alteração", "Restrição", "Tipo", "Quantidade solicitada", "Variação (%)",
                    "Δ", "LD anterior", "LD proposto",
                    "Viabilidade da base anterior", "Preço aplicado", "Z anterior", "ZNOVO",
                    "Base / resultado", "Método"])
        number = st.selectbox("Consultar cálculos da alteração",
                              range(1, len(session.history) + 1))
        entry = session.history[number - 1]
        if st.button("Restaurar estado selecionado", disabled=entry.result.status != "ótimo"):
            session.restore(number)
            st.rerun()
        show_change(problem, entry, context=f"history_{st.session_state['model_version']}_{number}")
