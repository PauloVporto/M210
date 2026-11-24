"""Conversão dos pedidos de alteração de recursos em novos LDs."""

from math import isfinite


def resource_rhs(current, operation, unit, value):
    """Usa o LD vigente como referência para uma única alteração."""
    if not isfinite(current) or not isfinite(value):
        raise ValueError("Informe valores finitos.")
    if operation in ("Definir", "Restaurar LD original"):
        return value
    if operation == "Dobrar":
        return resource_rhs(current, "Multiplicar", "vezes", 2)
    if operation == "Multiplicar":
        if unit != "vezes" or value < 0:
            raise ValueError("Informe um fator não negativo.")
        return resource_rhs(current, "Aumentar" if value >= 1 else "Diminuir", unit, value)
    if operation not in ("Aumentar", "Diminuir") or value < 0:
        raise ValueError("A quantidade de aumento ou diminuição deve ser não negativa.")
    direction = 1 if operation == "Aumentar" else -1
    if unit == "%":
        updated = current + direction * current * (value / 100)
    elif unit == "unidades":
        updated = current + direction * value
    elif unit == "vezes":
        if (direction > 0 and value < 1) or (direction < 0 and value > 1):
            raise ValueError("Use um multiplicador ≥ 1 para aumentar ou entre 0 e 1 para diminuir.")
        updated = current * value
    else:
        raise ValueError("Unidade de alteração inválida.")
    if not isfinite(updated):
        raise ValueError("O novo LD está fora do intervalo numérico.")
    return updated


def relative_percentage(current, delta):
    if current == 0:
        return None
    percentage = (delta / current) * 100
    return percentage if isfinite(percentage) else None


def operation_description(operation, unit, value, fmt):
    if operation == "Dobrar":
        return "Dobrar", "2 vezes"
    if operation == "Restaurar LD original":
        return operation, fmt(value)
    if operation == "Definir":
        return "Definir novo LD", fmt(value)
    return operation, f"{fmt(value)}{unit if unit == '%' else ' ' + unit}"


def operation_calculation(old, operation, unit, value, fmt):
    """Expressão de Δ a partir dos parâmetros efetivamente solicitados."""
    if operation in ("Definir", "Restaurar LD original"):
        return f"{fmt(value)} − ({fmt(old)})"
    if unit == "%":
        sign = "−" if operation == "Diminuir" else ""
        return f"{sign}({fmt(old)}) × {fmt(value)} / 100"
    if unit == "unidades":
        return ("−" if operation == "Diminuir" else "+") + fmt(value)
    return f"({fmt(old)}) × {fmt(value)} − ({fmt(old)})"
