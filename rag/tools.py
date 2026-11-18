"""Agent tools that don't need the LLM: calculator and table analysis."""
import ast
import math
import operator as op

import pandas as pd

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
_FUNCS = {"sqrt": math.sqrt, "round": round, "abs": abs, "min": min, "max": max, "sum": sum, "log": math.log}


def calculate(expression: str) -> str:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        raise ValueError("unsupported expression")
    try:
        result = ev(ast.parse(expression.strip(), mode="eval").body)
        return f"{expression} = {round(result, 6) if isinstance(result, float) else result}"
    except Exception as e:
        return f"Calculator error: {e}"


def analyze_data(df: pd.DataFrame, operation: str, column: str | None = None, group_by: str | None = None,
                 agg: str = "mean", n: int = 5, ascending: bool = False) -> str:
    def need(col):
        if col not in df.columns:
            raise ValueError(f"column '{col}' not found. Columns: {list(df.columns)}")
    if agg not in ("mean", "sum", "min", "max", "median", "count"):
        raise ValueError("agg must be one of: mean, sum, min, max, median, count")
    if operation == "describe":
        return df.describe(include="all").round(3).to_string()
    if operation == "head":
        return df.head(n).to_string(index=False)
    if operation == "value_counts":
        need(column)
        return df[column].value_counts().head(20).to_string()
    if operation == "stat":
        need(column)
        return f"{agg}({column}) = {getattr(df[column], agg)()}"
    if operation == "groupby":
        need(column); need(group_by)
        return df.groupby(group_by)[column].agg(agg).round(3).to_string()
    if operation == "top":
        need(column)
        return df.sort_values(column, ascending=ascending).head(n).to_string(index=False)
    raise ValueError("operation must be one of: describe, head, value_counts, stat, groupby, top")
