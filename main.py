"""
Simulador de convolución
    · continua   y(t) = ∫ x(τ)·h(t−τ) dτ
    · discreta   y[n] = Σ x[k]·h[n−k]

Estética "dashboard oscuro": fondo azul-negro con rejilla, tipografía monoespaciada,
tarjetas con borde fino, acentos violeta / rosa / verde y botones tipo píldora.

Pestañas: 1 señales · 2 reflexión · 3 animación · 4 resultado y(t) completo.
Selector CONTINUO / DISCRETO arriba a la izquierda.

Estabilidad: PyQt6 aborta el proceso si una excepción de Python escapa de un slot (la
ventana "se cierra sola"). Aquí hay un sys.excepthook que lo evita, los valores numéricos se
sanean (NaN/inf/enormes) y cualquier error queda en convolucion_error.log.
"""
from __future__ import annotations

import ast
import faulthandler
import json
import os
import re
import sys
import traceback

import matplotlib as mpl
import numpy as np
from PyQt6.QtCore import QPointF, QProcess, Qt, QTimer
from PyQt6.QtGui import (QBrush, QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPen,
                         QRadialGradient)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QDoubleSpinBox, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton,
                             QSlider,
                             QTabWidget, QVBoxLayout, QWidget)
from matplotlib import style as mpl_style
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------------------
# Registro de errores (evita que un fallo cierre el programa sin dejar rastro)
# --------------------------------------------------------------------------------------
_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "convolucion_error.log")
_FAULT_LOG = None  # se mantiene referenciado para que faulthandler no pierda el archivo


def _log(text: str) -> None:
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(text + ("\n" if not text.endswith("\n") else ""))
    except OSError:
        pass
    err = getattr(sys, "__stderr__", None)
    if err is not None:
        try:
            err.write(text)
        except Exception:  # noqa: BLE001
            pass


def _excepthook(exc_type, exc, tb):
    """Con esto instalado, PyQt6 NO aborta el proceso ante una excepción en un slot."""
    _log("".join(traceback.format_exception(exc_type, exc, tb)))


# --------------------------------------------------------------------------------------
# Paleta (tomada del dashboard de referencia)
# --------------------------------------------------------------------------------------
BG, CARD, CARD_EDGE = "#080b11", "#0e131c", "#1c2331"
INPUT_BG, PLOT_BG, GRID = "#0a0e15", "#090c12", "#1a2130"
TEXT, SOFT, MUTED = "#e5e7eb", "#9ca3af", "#6b7280"
ACCENT, ACCENT_SOFT = "#7c6cf0", "#a78bfa"
X_COL, H_COL = "#8b7cf6", "#f0589e"          # x violeta · h rosa
GREEN, GREEN_SOFT = "#22c55e", "#4ade80"
ORANGE, RED_SOFT = "#f59e0b", "#f87171"

MONO_FAMILIES = ["JetBrains Mono", "Cascadia Mono", "Consolas", "SF Mono", "Menlo", "Courier New"]
MONO_CSS = ", ".join(f"'{f}'" for f in MONO_FAMILIES) + ", monospace"

mpl_style.use("dark_background")
mpl.rcParams.update({
    "font.family": "monospace",
    "font.monospace": MONO_FAMILIES + ["DejaVu Sans Mono"],
    "font.size": 9,
    "figure.facecolor": CARD,
    "axes.facecolor": PLOT_BG,
    "axes.edgecolor": CARD_EDGE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": GRID,
})

# ===== PARSER START =====
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz  # compatible con NumPy 1.x y 2.x


class SignalError(ValueError):
    """Error de interpretación de una señal escrita por el usuario."""


_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
    ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.BitAnd, ast.BitOr,
    ast.USub, ast.UAdd, ast.Invert,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)

_CONSTANTS = {"pi": float(np.pi), "e": float(np.e)}
_VARS = ("t", "n", "k")      # alias de la misma variable independiente
_CLIP_SIGNAL = 1e9           # tope para que ningún valor llegue a inf ni rompa matplotlib


def _build_functions(dt: float, discrete: bool = False) -> dict:
    """Funciones disponibles dentro de las expresiones.

    En modo continuo `dt` define el ancho de la delta (triángulo de área 1).
    En modo discreto la delta es exacta: δ[n] = 1 solo en n = 0.
    """
    eps = 1e-9 if discrete else 0.0

    def arr(x):
        return np.asarray(x, dtype=float)

    def u(x):
        return (arr(x) >= -eps).astype(float)

    def ramp(x):
        return arr(x) * u(x)

    def rect(x):
        return (np.abs(arr(x)) <= 0.5 + eps).astype(float)

    def tri(x):
        return np.maximum(1.0 - np.abs(arr(x)), 0.0)

    if discrete:
        def delta(x):
            return (np.abs(arr(x)) < 0.5).astype(float)
    else:
        def delta(x):
            # Impulso discretizado de área 1: triángulo de ancho 2·dt y altura 1/dt
            return np.maximum(1.0 - np.abs(arr(x)) / dt, 0.0) / dt

    return {
        "u": u, "step": u, "heaviside": u,
        "ramp": ramp, "rect": rect, "tri": tri, "delta": delta, "sinc": np.sinc,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
        "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign, "sgn": np.sign,
        "floor": np.floor, "ceil": np.ceil,
        "max": np.maximum, "min": np.minimum, "where": np.where,
    }


class _Rewriter(ast.NodeTransformer):
    """Adapta la sintaxis de Python a operaciones vectorizadas de NumPy."""

    def visit_Constant(self, node):
        if type(node.value) is int:  # 9**9**9 con enteros podría colgar el programa
            return ast.copy_location(ast.Constant(float(node.value)), node)
        return node

    def visit_Compare(self, node):  # 0 <= t < 2  ->  (0 <= t) & (t < 2)
        self.generic_visit(node)
        if len(node.ops) == 1:
            return node
        parts, left = [], node.left
        for op, right in zip(node.ops, node.comparators):
            parts.append(ast.Compare(left=left, ops=[op], comparators=[right]))
            left = right
        out = parts[0]
        for p in parts[1:]:
            out = ast.BinOp(left=out, op=ast.BitAnd(), right=p)
        return out

    def visit_BoolOp(self, node):  # and / or  ->  & / |
        self.generic_visit(node)
        op = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        out = node.values[0]
        for v in node.values[1:]:
            out = ast.BinOp(left=out, op=op, right=v)
        return out

    def visit_UnaryOp(self, node):  # not  ->  ~
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node


_NOT_KW = r"(?!(?:and|or)\b)"


def _preprocess(expr: str) -> str:
    s = expr.strip()
    for a, b in (("·", "*"), ("×", "*"), ("−", "-"), ("τ", "t"), ("π", "pi"), ("δ", "delta"),
                 ("[", "("), ("]", ")"), ("^", "**")):
        s = s.replace(a, b)
    # Multiplicación implícita:  2t -> 2*t,  3sin(t) -> 3*sin(t),  (t+1)u(t) -> (t+1)*u(t)
    s = re.sub(r"(?<![\w.])(\d+\.?\d*|\.\d+)\s*(?![eE][+-]?\d)" + _NOT_KW + r"(?=[A-Za-z_(])",
               r"\1*", s)
    s = re.sub(r"\)\s*" + _NOT_KW + r"(?=[\w(.])", ")*", s)
    return s


def _sane(a, limit: float = 1e18):
    """Reemplaza NaN/±inf por 0 y acota la magnitud: evita errores al fijar límites y dibujar."""
    return np.clip(np.nan_to_num(np.asarray(a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0),
                   -limit, limit)


def _compile(expr: str, dt: float, discrete: bool = False):
    text = _preprocess(expr)
    if not text:
        raise SignalError("La expresión está vacía.")
    if len(text) > 400:
        raise SignalError("La expresión es demasiado larga.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise SignalError("Sintaxis inválida: revisa paréntesis y operadores.") from None

    tree = ast.fix_missing_locations(_Rewriter().visit(tree))
    funcs = _build_functions(dt, discrete)

    call_names = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SignalError(f"Elemento no permitido en la expresión ({type(node).__name__}).")
        if isinstance(node, ast.Constant) and type(node.value) is not float:
            raise SignalError("Solo se permiten números reales.")
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name) or node.func.id not in funcs:
                name = node.func.id if isinstance(node.func, ast.Name) else "?"
                raise SignalError(f"Función desconocida: «{name}».")
        elif isinstance(node, ast.Name) and id(node) not in call_names:
            if node.id not in _VARS and node.id not in _CONSTANTS:
                raise SignalError(f"Nombre desconocido: «{node.id}». La variable es "
                                  f"{'n' if discrete else 't'}.")

    code = compile(tree, "<señal>", "eval")
    env = {"__builtins__": {}, **funcs, **_CONSTANTS}

    def signal(t):
        t = np.asarray(t, dtype=float)
        try:
            with np.errstate(all="ignore"):
                res = np.asarray(eval(code, env, {"t": t, "n": t, "k": t}),  # noqa: S307
                                 dtype=float)
                res = np.broadcast_to(res, t.shape)
        except Exception as exc:
            raise SignalError(f"No se pudo evaluar la expresión ({exc}).") from None
        return _sane(res, _CLIP_SIGNAL)

    signal(np.array([-1.0, 0.0, 1.0]))  # prueba rápida: que los errores salgan al compilar
    return signal


def compile_signal(expr: str, dt: float, discrete: bool = False):
    """Devuelve f(t) -> ndarray vectorizada, o lanza SignalError con un mensaje legible."""
    try:
        return _compile(expr, dt, discrete)
    except SignalError:
        raise
    except Exception as exc:
        raise SignalError("No se pudo interpretar la expresión.") from exc
# ===== PARSER END =====

# ===== SYMBOLIC START =====
# Convolución EXACTA por tramos (continua: integral · discreta: sumatoria) con SymPy.
# Se corre en un proceso aparte (ver --symbolic) para no congelar la ventana y poder cortarlo.
class SymbolicError(Exception):
    """No se pudo obtener una fórmula cerrada (el motivo va en el mensaje)."""


def _sp():
    import sympy as sp
    return sp


def _to_sympy(expr: str, discrete: bool, X):
    """Traduce la expresión del usuario (misma sintaxis que el parser numérico) a SymPy."""
    sp = _sp()
    from sympy.logic.boolalg import Boolean
    text = _preprocess(expr)
    tree = ast.fix_missing_locations(_Rewriter().visit(ast.parse(text, mode="eval")))

    def H(a, at0=None):
        if discrete:
            return sp.Heaviside(a, 1 if at0 is None else at0)
        return sp.Heaviside(a)

    def ramp(a):
        return a * H(a)

    funcs = {
        "u": H, "step": H, "heaviside": H,
        "ramp": ramp,
        # rect = 1 si |a| <= 1/2 ; en discreto los extremos ±1/2 cuentan (igual que el cálculo numérico)
        "rect": lambda a: H(a + sp.Rational(1, 2), 1) - H(a - sp.Rational(1, 2), 0),
        "tri": lambda a: ramp(a + 1) - 2 * ramp(a) + ramp(a - 1),
        "delta": lambda a: sp.DiracDelta(a),
        "sinc": lambda a: sp.sin(sp.pi * a) / (sp.pi * a),
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
        "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        "exp": sp.exp, "log": sp.log, "ln": sp.log, "log10": lambda a: sp.log(a, 10),
        "sqrt": sp.sqrt, "abs": sp.Abs, "sign": sp.sign, "sgn": sp.sign,
        "floor": sp.floor, "ceil": sp.ceiling,
        "max": sp.Max, "min": sp.Min,
    }

    def is_bool(v):
        # OJO: en SymPy un Symbol también hereda de Boolean; lo que cuenta es que NO sea Expr
        return isinstance(v, Boolean) and not isinstance(v, sp.Expr)

    def num(v):
        """Un booleano usado en una cuenta vale 1 / 0."""
        return sp.Piecewise((1, v), (0, True)) if is_bool(v) else v

    def boolean(v):
        return v if is_bool(v) else sp.Ne(v, 0)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return sp.Rational(repr(float(n.value)))
        if isinstance(n, ast.Name):
            if n.id in _VARS:
                return X
            return {"pi": sp.pi, "e": sp.E}[n.id]
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.USub):
                return -num(v)
            if isinstance(n.op, ast.UAdd):
                return num(v)
            return sp.Not(boolean(v))
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.BitAnd):
                return sp.And(boolean(a), boolean(b))
            if isinstance(n.op, ast.BitOr):
                return sp.Or(boolean(a), boolean(b))
            a, b = num(a), num(b)
            return {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                    ast.Div: lambda: a / b, ast.Pow: lambda: a ** b,
                    ast.Mod: lambda: sp.Mod(a, b)}[type(n.op)]()
        if isinstance(n, ast.Compare):
            a, b = ev(n.left), ev(n.comparators[0])
            return {ast.Lt: sp.Lt, ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge,
                    ast.Eq: sp.Eq, ast.NotEq: sp.Ne}[type(n.ops[0])](num(a), num(b))
        if isinstance(n, ast.Call):
            name = n.func.id
            args = [ev(a) for a in n.args]
            if name == "where":
                return sp.Piecewise((num(args[1]), boolean(args[0])), (num(args[2]), True))
            return funcs[name](*[num(a) for a in args])
        raise SymbolicError("elemento no soportado en el modo simbólico")

    try:
        return ev(tree)
    except SymbolicError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SymbolicError(f"no se pudo traducir la expresión ({exc})") from None


def _freeze(expr, X, v0):
    """Congela Heaviside/Abs/Piecewise/Max/Min según el valor de X = v0 → expresión suave."""
    sp = _sp()

    def val(e):
        return e.subs(X, v0)

    def rec(e):
        if isinstance(e, sp.Heaviside):
            a = val(rec(e.args[0]))
            h0 = e.args[1] if len(e.args) > 1 else sp.Rational(1, 2)
            return sp.Integer(1) if a > 0 else (sp.Integer(0) if a < 0 else sp.sympify(h0))
        if isinstance(e, sp.Abs):
            a = rec(e.args[0])
            return a if val(a) >= 0 else -a
        if isinstance(e, sp.sign):
            v = val(rec(e.args[0]))
            return sp.Integer(1 if v > 0 else (-1 if v < 0 else 0))
        if isinstance(e, (sp.Max, sp.Min)):
            args = [rec(a) for a in e.args]
            pick = max if isinstance(e, sp.Max) else min
            return pick(args, key=lambda a: float(val(a)))
        if isinstance(e, sp.Piecewise):
            for ex, cond in e.args:
                cv = cond.subs(X, v0)
                if cv == True:  # noqa: E712
                    return rec(ex)
                if cv != False:  # noqa: E712
                    raise SymbolicError("condición no evaluable")
            return sp.Integer(0)
        if not e.args:
            return e
        return e.func(*[rec(a) for a in e.args])

    return rec(expr)


def _roots_of(arg, X):
    sp = _sp()
    if not arg.has(X):
        return []
    try:
        poly = sp.Poly(arg, X)
        if poly.degree() == 1:
            c1, c0 = poly.all_coeffs()
            return [-c0 / c1]
        if poly.degree() == 0:
            return []
    except sp.PolynomialError:
        pass
    try:
        sols = sp.solve(arg, X)
    except Exception:  # noqa: BLE001
        raise SymbolicError("no se pudieron hallar los puntos de cambio de la señal") from None
    return [s for s in sols if s.is_real is not False and s.is_number]


def _breakpoints(expr, X):
    sp = _sp()
    args = []
    for h in expr.atoms(sp.Heaviside):
        args.append(h.args[0])
    for a in expr.atoms(sp.Abs):
        args.append(a.args[0])
    for s in expr.atoms(sp.sign):
        args.append(s.args[0])
    for m in expr.atoms(sp.Max, sp.Min):
        args += [m.args[0] - a for a in m.args[1:]]
    for p in expr.atoms(sp.Piecewise):
        for _ex, cond in p.args:
            for rel in cond.atoms(sp.Rel):
                args.append(rel.lhs - rel.rhs)
    if expr.atoms(sp.floor, sp.ceiling):
        raise SymbolicError("floor/ceil no están soportados en el modo simbólico")
    roots = {}
    for a in args:
        for r in _roots_of(a, X):
            roots[float(r)] = r
    return [roots[k] for k in sorted(roots)]


def _decompose(expr, X, discrete):
    """Señal → (tramos suaves [(a, b, f(X))], impulsos [(posición, peso)])."""
    sp = _sp()
    oo = sp.oo
    expr = sp.expand(expr)
    smooth_terms, impulses = [], []
    for term in sp.Add.make_args(expr):
        if not term.has(sp.DiracDelta):
            smooth_terms.append(term)
            continue
        deltas = [f for f in sp.Mul.make_args(term) if isinstance(f, sp.DiracDelta)]
        if len(deltas) != 1:
            raise SymbolicError("producto de impulsos no soportado")
        d = deltas[0]
        coeff = sp.Mul(*[f for f in sp.Mul.make_args(term) if f is not d])
        if coeff.has(sp.DiracDelta):
            raise SymbolicError("producto de impulsos no soportado")
        poly = sp.Poly(d.args[0], X)
        if poly.degree() != 1:
            raise SymbolicError("delta con argumento no lineal no soportada")
        c1, c0 = poly.all_coeffs()
        r = -c0 / c1
        if discrete:
            if abs(c1) != 1:
                raise SymbolicError("delta discreta con escala distinta de 1 no soportada")
            if r.is_integer is not True:
                continue                      # no cae sobre ningún entero: vale 0 en la red
            w = _freeze(coeff, X, r).subs(X, r)
        else:
            w = _freeze(coeff, X, r).subs(X, r) / abs(c1)
        impulses.append((r, sp.nsimplify(w)))
    smooth = sp.Add(*smooth_terms)
    roots = _breakpoints(smooth, X)

    pieces = []

    def add(lo, hi, test):
        e = _freeze(smooth, X, test)
        e = sp.simplify(e)
        if e != 0:
            pieces.append((lo, hi, e))

    if discrete:
        lo = -oo
        spans = []
        for r in roots:
            spans.append((lo, sp.ceiling(r) - 1))
            if r.is_integer:
                spans.append((r, r))
            lo = sp.floor(r) + 1
        spans.append((lo, oo))
        for a, b in spans:
            if a.is_finite and b.is_finite and a > b:
                continue
            add(a, b, a if a.is_finite else (b if b.is_finite else sp.Integer(0)))
    else:
        pts = [-oo] + roots + [oo]
        for a, b in zip(pts[:-1], pts[1:]):
            if a.is_finite and b.is_finite:
                t0 = (a + b) / 2
            elif a.is_finite:
                t0 = a + 1
            elif b.is_finite:
                t0 = b - 1
            else:
                t0 = sp.Integer(0)
            add(a, b, t0)
    return pieces, impulses


def _pretty(e):
    """Elige, entre varias formas equivalentes, la más corta."""
    sp = _sp()
    cands = [e]
    for f in (sp.expand, lambda z: sp.powsimp(sp.expand(z)),
              lambda z: sp.powsimp(sp.expand(z), combine='base'), sp.simplify,
              lambda z: sp.expand(sp.simplify(z)), lambda z: sp.powsimp(sp.expand(sp.simplify(z))),
              lambda z: sp.powsimp(sp.expand(sp.simplify(z)), combine='base'), sp.factor):
        try:
            cands.append(f(e))
        except Exception:  # noqa: BLE001
            pass
    good = [c for c in cands if c == e or sp.simplify(c - e) == 0]
    return min(good, key=lambda c: (sp.count_ops(c), len(str(c)))) if good else e


def _fmt_expr(e):
    s = str(e).replace("**", "^")
    s = re.sub(r"\bpi\b", "π", s)
    s = re.sub(r"\bE\b", "e", s)
    return s


def _finite_sequence(xp, xq, hp, hq, T, X, discrete):
    """Respaldo discreto: si ambas señales son de duración finita, suma directa y exacta."""
    sp = _sp()
    if not discrete or not xp or not hp:
        return None
    if not all(a.is_finite and b.is_finite for a, b, _ in xp + hp) or xq or hq:
        return None
    n0 = min(a for a, _, _ in xp) + min(a for a, _, _ in hp)
    n1 = max(b for _, b, _ in xp) + max(b for _, b, _ in hp)
    if n1 - n0 > 200:
        return None
    out = []
    for nn in range(int(n0), int(n1) + 1):
        v = sp.Integer(0)
        for ai, bi, fi in xp:
            for aj, bj, gj in hp:
                for k in range(int(max(ai, nn - bj)), int(min(bi, nn - aj)) + 1):
                    v += fi.subs(X, k) * gj.subs(X, nn - k)
        v = sp.nsimplify(sp.simplify(v))
        out.append((sp.Integer(nn), sp.Integer(nn), v))
    return [r for r in out if r[2] != 0]


def symbolic_convolution(x_expr: str, h_expr: str, discrete: bool, R: float = 6.0) -> dict:
    """Devuelve {'ok', 'text', 'verified', ...}. Lanza SymbolicError si no hay fórmula cerrada."""
    sp = _sp()
    oo = sp.oo
    X = sp.Symbol("x_", real=True)
    S = sp.Symbol("s_", integer=True) if discrete else sp.Symbol("s_", real=True)
    T = sp.Symbol("n" if discrete else "t", integer=discrete, real=True)

    xp, xq = _decompose(_to_sympy(x_expr, discrete, X), X, discrete)
    hp, hq = _decompose(_to_sympy(h_expr, discrete, X), X, discrete)
    if not (xp or xq) or not (hp or hq):
        return {"ok": True, "verified": True, "text": f"{'y[n]' if discrete else 'y(t)'} = 0",
                "pieces": [], "impulses": [], "sequence": None}

    # ---- puntos de cambio de la salida
    cands = set()
    fin = lambda v: v.is_finite  # noqa: E731
    for ai, bi, _ in xp:
        for aj, bj, _ in hp:
            for u in (ai, bi):
                for v in (aj, bj):
                    if fin(u) and fin(v):
                        cands.add(u + v)
    for tk, _ in xq:
        for aj, bj, _ in hp:
            for v in (aj, bj):
                if fin(v):
                    cands.add(tk + v)
    for tj, _ in hq:
        for ai, bi, _ in xp:
            for v in (ai, bi):
                if fin(v):
                    cands.add(tj + v)
    for tk, _ in xq:
        for tj, _ in hq:
            cands.add(tk + tj)
    if discrete:
        cands |= {c + 1 for c in cands}
    cs = sorted(cands, key=float)

    if discrete:
        spans = []
        if not cs:
            spans.append((-oo, oo, sp.Integer(0)))
        else:
            spans.append((-oo, cs[0] - 1, cs[0] - 1))
            for c0, c1 in zip(cs[:-1], cs[1:]):
                spans.append((c0, c1 - 1, c0))
            spans.append((cs[-1], oo, cs[-1]))
    else:
        pts = [-oo] + cs + [oo]
        spans = []
        for a, b in zip(pts[:-1], pts[1:]):
            if a.is_finite and b.is_finite:
                t0 = (a + b) / 2
            elif a.is_finite:
                t0 = a + 1
            elif b.is_finite:
                t0 = b - 1
            else:
                t0 = sp.Integer(0)
            spans.append((a, b, t0))

    def inside(a, b, v):          # ¿v dentro del tramo (a, b) de la señal?
        return (a <= v <= b) if discrete else (a < v < b)

    sequence_only = False
    try:
        results = []                  # (lo, hi, expr)
        for lo_t, hi_t, t0 in spans:
            total = sp.Integer(0)
            for ai, bi, fi in xp:
                for aj, bj, gj in hp:
                    lo_opts = [(ai, ai)]
                    if bj != oo:
                        lo_opts.append((t0 - bj, T - bj))
                    hi_opts = [(bi, bi)]
                    if aj != -oo:
                        hi_opts.append((t0 - aj, T - aj))
                    lo_v, lo = max(lo_opts, key=lambda p: float(p[0]) if p[0].is_finite else -1e300)
                    hi_v, hi = min(hi_opts, key=lambda p: float(p[0]) if p[0].is_finite else 1e300)
                    if (hi_v < lo_v) if discrete else (hi_v <= lo_v):
                        continue
                    integrand = fi.subs(X, S) * gj.subs(X, T - S)
                    if discrete:
                        part = sp.summation(integrand, (S, lo, hi))
                    else:
                        part = sp.integrate(integrand, (S, lo, hi))
                    if part.has(sp.Sum) or part.has(sp.Integral) or part.has(sp.nan):
                        raise SymbolicError("SymPy no pudo resolver el cálculo en forma cerrada")
                    if part in (oo, -oo) or part.has(oo) or part.has(-oo):
                        raise SymbolicError("la convolución diverge (no existe en forma cerrada finita)")
                    total += part
            for tk, w in xq:
                for aj, bj, gj in hp:
                    if inside(aj + tk, bj + tk, t0):
                        total += w * gj.subs(X, T - tk)
            for tj, w in hq:
                for ai, bi, fi in xp:
                    if inside(ai + tj, bi + tj, t0):
                        total += w * fi.subs(X, T - tj)
            if discrete:
                for tk, wk in xq:
                    for tj, wj in hq:
                        if tk + tj == t0:
                            total += wk * wj
            total = _pretty(sp.simplify(total))
            if total != 0:
                results.append((lo_t, hi_t, total))

    except SymbolicError as exc:
        fb = _finite_sequence(xp, xq, hp, hq, T, X, discrete) if discrete else None
        if fb is None or 'diverge' in str(exc):
            raise
        results, sequence_only = fb, True

    # ---- (discreto) un punto aislado cuyo valor ya da la fórmula vecina se absorbe en ella
    if discrete:
        changed = True
        while changed:
            changed = False
            for idx, (lo, hi, e) in enumerate(results):
                if lo != hi or not lo.is_finite:
                    continue
                v = e.subs(T, lo)
                if idx > 0 and results[idx - 1][1] + 1 == lo and \
                        sp.simplify(results[idx - 1][2].subs(T, lo) - v) == 0:
                    results[idx - 1] = (results[idx - 1][0], lo, results[idx - 1][2])
                    del results[idx]
                    changed = True
                    break
                if idx + 1 < len(results) and results[idx + 1][0] - 1 == lo and \
                        sp.simplify(results[idx + 1][2].subs(T, lo) - v) == 0:
                    results[idx + 1] = (lo, results[idx + 1][1], results[idx + 1][2])
                    del results[idx]
                    changed = True
                    break

    # ---- unir tramos contiguos con la misma fórmula
    merged = []
    for lo, hi, e in results:
        if merged:
            plo, phi, pe = merged[-1]
            contiguous = (phi + 1 == lo) if discrete else (phi == lo)
            if contiguous and sp.simplify(pe - e) == 0:
                merged[-1] = (plo, hi, pe)
                continue
        merged.append((lo, hi, e))

    cont_impulses = []
    if not discrete:
        for tk, wk in xq:
            for tj, wj in hq:
                cont_impulses.append((tk + tj, wk * wj))

    out = _verify_and_format(merged, cont_impulses, T, discrete, x_expr, h_expr, R, sequence_only)
    return out


def _verify_and_format(pieces, impulses, T, discrete, x_expr, h_expr, R, sequence_only=False):
    sp = _sp()
    name = "y[n]" if discrete else "y(t)"
    var = "n" if discrete else "t"

    def cond_text(lo, hi):
        if discrete:
            if lo == hi:
                return f"{var} = {lo}"
            if lo == -sp.oo:
                return f"{var} ≤ {hi}"
            if hi == sp.oo:
                return f"{var} ≥ {lo}"
            return f"{lo} ≤ {var} ≤ {hi}"
        if lo == -sp.oo:
            return f"{var} < {_fmt_expr(hi)}"
        if hi == sp.oo:
            return f"{var} ≥ {_fmt_expr(lo)}"
        return f"{_fmt_expr(lo)} ≤ {var} < {_fmt_expr(hi)}"

    # ---------- verificación numérica contra el cálculo normal (el de las gráficas)
    R = float(R)
    if discrete:
        Ri = int(round(R))
        k = np.arange(-2 * Ri, 2 * Ri + 1, dtype=float)
        fx = compile_signal(x_expr, 1.0, True)
        fh = compile_signal(h_expr, 1.0, True)
        pts = np.arange(-Ri, Ri + 1, dtype=float)
        ref = np.array([float(np.sum(fx(k) * fh(p - k))) for p in pts])
    else:
        dt = 0.004
        k = np.arange(-2 * R, 2 * R + dt / 2, dt)
        fx = compile_signal(x_expr, dt, False)
        fh = compile_signal(h_expr, dt, False)
        pts = np.linspace(-R + 0.0137, R - 0.0171, 61)
        near = [float(c) for _, hi_, _ in pieces for c in (hi_,) if c.is_finite]
        near += [float(lo_) for lo_, _, _ in pieces if lo_.is_finite]
        near += [float(p) for p, _ in impulses]
        pts = np.array([p for p in pts if all(abs(p - c) > 4 * dt for c in near)])
        xv = fx(k)
        ref = np.array([float(_TRAPZ(xv * fh(p - k), k)) for p in pts])

    fns = [(lo, hi, sp.lambdify(T, e, "mpmath")) for lo, hi, e in pieces]
    approx = np.zeros_like(ref)
    for idx, p in enumerate(pts):
        for lo, hi, f in fns:
            lo_ok = lo == -sp.oo or p >= float(lo)
            hi_ok = hi == sp.oo or (p <= float(hi) if discrete else p < float(hi))
            if lo_ok and hi_ok:
                approx[idx] += complex(f(p)).real
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    tol = (1e-7 if discrete else 4e-2) * scale
    verified = bool(ref.size) and float(np.max(np.abs(approx - ref))) <= tol

    # ---------- texto
    if sequence_only:
        body = "No hay una fórmula cerrada simple (SymPy no pudo sumar la expresión)."
    elif not pieces and not impulses:
        body = f"{name} = 0"
    else:
        lines = []
        for lo, hi, e in pieces:
            lines.append((_fmt_expr(e), cond_text(lo, hi)))
        for p, w in impulses:
            wtxt = "" if w == 1 else f"{_fmt_expr(w)}·"
            lines.append((f"{wtxt}δ({var} − {_fmt_expr(p)})" if p != 0 else f"{wtxt}δ({var})",
                          "impulso"))
        width = max(len(a) for a, _ in lines)
        if len(lines) == 1 and not impulses and pieces[0][0] == -sp.oo and pieces[0][1] == sp.oo:
            body = f"{name} = {lines[0][0]}"
        else:
            body = f"{name} =\n" + "\n".join(f"   {a.ljust(width)}   ,  {c}" for a, c in lines)
            body += "\n   0" + " " * max(0, width - 1) + "   ,  en el resto"

    sequence = None
    if discrete and pieces and all(lo.is_finite and hi.is_finite for lo, hi, _ in pieces):
        a0 = min(int(lo) for lo, _, _ in pieces)
        b0 = max(int(hi) for _, hi, _ in pieces)
        if b0 - a0 <= 80:
            vals = []
            for nn in range(a0, b0 + 1):
                v = sp.Integer(0)
                for lo, hi, e in pieces:
                    if lo <= nn <= hi:
                        v += e.subs(T, nn)
                vals.append(sp.nsimplify(v))
            sequence = {"start": a0, "values": [str(v) for v in vals]}
            body += (("\n\n" if body else "") + f"Valores exactos (desde n = {a0}):  y[n] = {{ "
                     + ", ".join(str(v) for v in vals) + " }")

    return {"ok": True, "verified": verified, "text": body,
            "pieces": [(cond_text(lo, hi), _fmt_expr(e)) for lo, hi, e in pieces],
            "impulses": [(_fmt_expr(p), _fmt_expr(w)) for p, w in impulses],
            "sequence": sequence}


def _symbolic_cli(argv):
    """python main.py --symbolic <cont|disc> <x> <h> <R>  →  JSON por stdout."""
    import json
    try:
        mode, x_expr, h_expr, R = argv[0], argv[1], argv[2], float(argv[3])
        res = symbolic_convolution(x_expr, h_expr, mode == "disc", R)
        if not res["verified"]:
            res = {"ok": False, "reason": "La fórmula obtenida no coincide con el cálculo numérico, "
                                           "así que no se muestra para no darte un resultado falso."}
    except SymbolicError as exc:
        res = {"ok": False, "reason": str(exc)}
    except Exception as exc:  # noqa: BLE001
        res = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(res, ensure_ascii=True))
    sys.stdout.flush()
# ===== SYMBOLIC END =====


# --------------------------------------------------------------------------------------
# Constantes de la interfaz
# --------------------------------------------------------------------------------------
N_FRAMES = 300            # frames de la animación continua (1× = 1 frame por tick)
DISCRETE_STEP = 0.05      # frames por tick a 1× en discreto (hay pocos frames: uno por cada n)
TICK_MS = 25              # 40 ticks por segundo
DEBOUNCE_MS = 250         # espera tras escribir antes de recalcular
SYMBOLIC_TIMEOUT_MS = 60000  # tiempo máximo para la fórmula exacta
SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# (nombre corto, x(t), h(t))
PRESETS = [
    ("pulso ∗ exp", "u(t) - u(t-2)", "exp(-t) * u(t)"),
    ("pulso ∗ pulso", "u(t) - u(t-2)", "u(t) - u(t-2)"),
    ("exp ∗ exp", "exp(-t) * u(t)", "exp(-2*t) * u(t)"),
    ("rampa ∗ pulso", "t * u(t)", "u(t) - u(t-1)"),
    ("seno ∗ exp", "sin(2*pi*t) * (u(t) - u(t-2))", "exp(-t) * u(t)"),
    ("tri ∗ rect", "tri(t)", "rect(t)"),
    ("delta ∗ exp", "delta(t-1)", "exp(-t) * u(t)"),
    ("por tramos", "(0 <= t < 1)*t + (1 <= t < 2)*(2 - t)", "rect(t - 0.5)"),
]

# (nombre corto, x[n], h[n])
DISCRETE_PRESETS = [
    ("rect ∗ rect", "u(n) - u(n-4)", "u(n) - u(n-4)"),
    ("escalón ∗ exp", "u(n)", "0.5^n * u(n)"),
    ("exp ∗ exp", "0.8^n * u(n)", "0.5^n * u(n)"),
    ("rampa ∗ pulso", "ramp(n) * (u(n) - u(n-5))", "u(n) - u(n-3)"),
    ("seno ∗ media móvil", "sin(pi*n/4) * (u(n) - u(n-8))", "(u(n) - u(n-3)) / 3"),
    ("δ desplazada", "delta(n-2)", "0.7^n * u(n)"),
    ("tri ∗ 2δ", "tri(n/3)", "delta(n) + delta(n-4)"),
    ("por tramos", "(0 <= n < 3)*(n+1) + (3 <= n < 6)*(6 - n)", "rect((n-1)/2)"),
]

HELP_CONT = (
    "Variable t (o τ) · constantes pi, e · potencia t^2\n"
    "u(t) ramp rect tri sinc delta · sin cos tan exp log sqrt abs sign · where(c,a,b) max min\n"
    "Comparaciones y lógica: (0 <= t < 2), and, or, not\n"
    "El * es opcional: 2t, 3sin(t), (t+1)u(t)"
)
HELP_DISC = (
    "Variable n (o k) · constantes pi, e · potencia 0.5^n\n"
    "u(n) delta(n) ramp rect tri · sin cos exp log sqrt abs sign · where(c,a,b) max min\n"
    "Solo se evalúa en enteros: delta(n-2) vale 1 en n = 2 y 0 en el resto\n"
    "Corchetes válidos: u[n], delta[n-2]"
)


# Ayudas
def fn(discrete: bool, name: str, arg: str) -> str:
    """x(t) en continuo, x[n] en discreto."""
    return f"{name}[{arg}]" if discrete else f"{name}({arg})"


def fmt_value(v: float) -> str:
    return f"{v:+.3f}" if abs(v) < 1e4 else f"{v:+.3e}"


def _limits(*arrays, pad: float = 0.12):
    """Límites Y que incluyen el 0 y todos los arreglos dados, con margen."""
    lo = min([0.0] + [float(np.min(a)) for a in arrays if np.size(a)])
    hi = max([0.0] + [float(np.max(a)) for a in arrays if np.size(a)])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return -1.0, 1.0
    span = (hi - lo) or 1.0
    return lo - pad * span, hi + pad * span


def _segs(xs, ys):
    """Segmentos verticales (x, 0) → (x, y) para dibujar stems con un LineCollection."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return np.empty((0, 2, 2))
    z = np.zeros_like(ys)
    return np.stack([np.column_stack([xs, z]), np.column_stack([xs, ys])], axis=1)


def _make_stem(ax, color, lw=2.0, alpha=1.0, ms=6, zorder=3):
    """Stem reutilizable: LineCollection para los palitos + línea de marcadores."""
    lc = LineCollection([], colors=color, linewidths=lw, alpha=alpha, zorder=zorder)
    ax.add_collection(lc, autolim=False)
    (mk,) = ax.plot([], [], linestyle="none", marker="o", markersize=ms, color=color,
                    alpha=alpha, zorder=zorder + 0.1)
    return lc, mk


def _set_stem(lc, mk, xs, ys):
    lc.set_segments(_segs(xs, ys))
    mk.set_data(xs, ys)


GRID_PX = 28              # paso de la rejilla del fondo

_QSS = """
QWidget { background: transparent; color: %(TEXT)s; font-family: %(MONO_CSS)s; font-size: 9pt; }
QMainWindow { background-color: %(BG)s; }
QToolTip { background-color: %(CARD)s; color: %(TEXT)s; border: 1px solid %(CARD_EDGE)s; }

QTabWidget::pane { border: none; background: transparent; }
QTabBar { background: transparent; }
QTabBar::tab {
    background: transparent; color: %(MUTED)s; padding: 9px 18px; margin-right: 8px;
    border: none; border-bottom: 2px solid transparent; font-weight: bold; font-size: 8pt;
}
QTabBar::tab:hover { color: %(SOFT)s; }
QTabBar::tab:selected { color: %(ACCENT_SOFT)s; border-bottom: 2px solid %(ACCENT)s; }

QFrame#card { background-color: %(CARD)s; border: 1px solid %(CARD_EDGE)s; border-radius: 14px; }
QLabel { background: transparent; }
QLabel#badge {
    background-color: #0b2a1a; border: 1px solid #1b7f45; border-radius: 15px;
    color: %(GREEN_SOFT)s; padding: 7px 16px; font-weight: bold; font-size: 8pt;
}
QLabel#error { color: %(RED_SOFT)s; font-size: 8pt; }
QLabel#speed { color: %(ORANGE)s; font-weight: bold; min-width: 112px; }

/* Botón base: píldora gris (como los chips "SiliconFlow", "Gemini"...) */
QPushButton {
    background-color: #131923; color: %(SOFT)s; border: 1px solid #232b3a;
    border-radius: 14px; padding: 7px 16px; font-weight: bold; font-size: 8pt;
}
QPushButton:hover { border-color: %(ACCENT)s; color: #c4b5fd; }
QPushButton:pressed { background-color: %(BG)s; }
QPushButton:disabled { color: #3b4252; border-color: #1a2130; background-color: #0f141d; }

/* Primario y chips: píldora verde (como "vertex 300M", "agentrouter 200M"...) */
QPushButton[kind="primary"], QPushButton[chip="true"] {
    background-color: #0b2a1a; color: %(GREEN_SOFT)s; border: 1px solid #1b7f45;
}
QPushButton[kind="primary"]:hover, QPushButton[chip="true"]:hover {
    background-color: #0f3a24; border-color: %(GREEN)s; color: #86efac;
}
QPushButton[kind="primary"]:pressed, QPushButton[chip="true"]:pressed { background-color: #07180f; }
QPushButton[chip="true"] { padding: 5px 12px; border-radius: 12px; }

/* Selector de modo CONTINUO / DISCRETO: gris, y verde el activo */
QPushButton[seg="true"]:checked {
    background-color: #0b2a1a; color: %(GREEN_SOFT)s; border: 1px solid #1b7f45;
}

QLineEdit, QDoubleSpinBox {
    background-color: %(INPUT_BG)s; color: %(TEXT)s; border: 1px solid #1d2431;
    border-radius: 8px; padding: 7px 10px; selection-background-color: %(ACCENT)s;
}
QLineEdit:focus, QDoubleSpinBox:focus { border: 1px solid %(ACCENT)s; }

QPlainTextEdit {
    background-color: %(INPUT_BG)s; color: %(TEXT)s; border: 1px solid #1d2431;
    border-radius: 8px; padding: 8px; font-size: 10pt;
}

QSlider::groove:horizontal { height: 8px; background: #161c27; border-radius: 4px; }
QSlider::sub-page:horizontal { background: #6d5bf0; border-radius: 4px; }
QSlider::handle:horizontal {
    background: #f3f4f6; width: 10px; height: 10px; margin: -1px 0; border-radius: 5px;
}
"""
DARK_STYLE = _QSS % {
    "TEXT": TEXT, "SOFT": SOFT, "MUTED": MUTED, "BG": BG, "CARD": CARD, "CARD_EDGE": CARD_EDGE,
    "INPUT_BG": INPUT_BG, "ACCENT": ACCENT, "ACCENT_SOFT": ACCENT_SOFT, "GREEN": GREEN,
    "GREEN_SOFT": GREEN_SOFT, "ORANGE": ORANGE, "RED_SOFT": RED_SOFT, "MONO_CSS": MONO_CSS,
}


# --------------------------------------------------------------------------------------
# Widgets y ayudas de estilo
# --------------------------------------------------------------------------------------
class Backdrop(QWidget):
    """Fondo: azul-negro con rejilla tenue y resplandor rosado arriba a la izquierda."""

    def paintEvent(self, event):
        p = QPainter(self)
        r = event.rect()
        p.fillRect(r, QColor(BG))

        glow = QRadialGradient(QPointF(190, 210), 380)
        glow.setColorAt(0.0, QColor(200, 55, 100, 70))
        glow.setColorAt(1.0, QColor(200, 55, 100, 0))
        p.fillRect(r, QBrush(glow))

        p.setPen(QPen(QColor(255, 255, 255, 9), 1))
        for x in range((r.left() // GRID_PX) * GRID_PX, r.right() + 1, GRID_PX):
            p.drawLine(x, r.top(), x, r.bottom())
        for y in range((r.top() // GRID_PX) * GRID_PX, r.bottom() + 1, GRID_PX):
            p.drawLine(r.left(), y, r.right(), y)


class GradientLabel(QLabel):
    """Texto grande con degradado rosa → violeta (como el '~1.62B' del dashboard)."""

    def __init__(self, text="", point_size=44):
        super().__init__(text)
        self._font = QFont()
        self._font.setFamilies(MONO_FAMILIES)
        self._font.setPointSize(point_size)
        self._font.setBold(True)
        fm = QFontMetrics(self._font)
        self.setMinimumHeight(fm.height() + 6)
        self.setMinimumWidth(fm.horizontalAdvance("+0.000e+00") + 10)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setFont(self._font)
        width = max(QFontMetrics(self._font).horizontalAdvance(self.text()), 1)
        grad = QLinearGradient(0, 0, width, 0)
        grad.setColorAt(0.0, QColor("#ec4899"))
        grad.setColorAt(1.0, QColor("#a855f7"))
        p.setPen(QPen(QBrush(grad), 1))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self.text())


def caption(html, color=MUTED, size=8, spacing=1.8, bold=True):
    """Etiqueta pequeña, en mayúsculas y con espaciado entre letras."""
    lbl = QLabel(html)
    font = lbl.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    lbl.setFont(font)
    lbl.setStyleSheet(f"color: {color}; font-size: {size}pt; "
                      f"font-weight: {'bold' if bold else 'normal'};")
    return lbl


def make_card(title=None, horizontal=False, margins=(18, 16, 18, 16)):
    frame = QFrame()
    frame.setObjectName("card")
    lay = QHBoxLayout(frame) if horizontal else QVBoxLayout(frame)
    lay.setContentsMargins(*margins)
    lay.setSpacing(12 if horizontal else 10)
    if title:
        lay.addWidget(caption(title, ACCENT_SOFT, spacing=2.2))
    return frame, lay


def make_button(text, kind=None, chip=False, seg=False):
    btn = QPushButton(text)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    if kind:
        btn.setProperty("kind", kind)
    if chip:
        btn.setProperty("chip", True)
    if seg:
        btn.setProperty("seg", True)
        btn.setCheckable(True)
    return btn


def dot_html(color, text):
    return f"<span style='color:{color}'>●</span>&nbsp;<span style='color:{SOFT}'>{text}</span>"


def _style_axis(ax, title=None, integer_x=False):
    ax.set_facecolor(PLOT_BG)
    ax.axhline(0, color="#2a3243", linewidth=1)
    ax.axvline(0, color="#2a3243", linewidth=0.8, linestyle="--")
    ax.grid(True, color=GRID, linestyle=":")
    for spine in ax.spines.values():
        spine.set_color(CARD_EDGE)
    ax.tick_params(colors=MUTED, labelsize=8)
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        ax.set_title(title, loc="left", color=MUTED, fontsize=8, fontweight="bold", pad=10)


def _legend(ax):
    ax.legend(loc="upper right", facecolor=CARD, edgecolor=CARD_EDGE, labelcolor=TEXT, fontsize=8)


def _new_figure(size):
    fig = Figure(figsize=size, facecolor=CARD)
    return fig, FigureCanvas(fig)



# --------------------------------------------------------------------------------------
# Ventana principal
# --------------------------------------------------------------------------------------
class ConvolutionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1280, 920)
        self.setStyleSheet(DARK_STYLE)

        # Modo elegido en la interfaz (self.discrete) y modo de los datos mostrados (self._d).
        # Difieren solo si, tras cambiar de modo, la expresión es inválida.
        self.discrete = False
        self._d = False

        # Datos calculados (se llenan en _rebuild)
        self.R = 6.0
        self.n_frames = N_FRAMES
        self.tau = self.t_frames = None
        self.x_vals = self.h_vals = self.h_ref = None
        self.H = self.P = self.y_frames = None
        self.sl = slice(0, 0)

        # Estado de la animación
        self.is_playing = False
        self.speed_idx = SPEEDS.index(1.0)
        self._pos = 0.0
        self._shown = -1
        self._fill = None

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)

        self.rebuild_timer = QTimer(self)
        self.rebuild_timer.setSingleShot(True)
        self.rebuild_timer.setInterval(DEBOUNCE_MS)
        self.rebuild_timer.timeout.connect(self._rebuild)

        # Fórmula exacta (SymPy) en un proceso aparte, con tiempo límite
        self.fx = self.fh = None
        self._sym_key = None
        self._sym_proc = None
        self.sym_timeout = QTimer(self)
        self.sym_timeout.setSingleShot(True)
        self.sym_timeout.setInterval(SYMBOLIC_TIMEOUT_MS)
        self.sym_timeout.timeout.connect(self._on_symbolic_timeout)

        self._build_ui()
        self._apply_mode_texts()
        self._update_speed_ui()
        self._update_live_ui()
        self._rebuild()

    # ------------------------------------------------------------------ UI
    def _schedule_rebuild(self, *_args):
        """Reinicia el debounce sin pasarle a QTimer.start los argumentos de la señal."""
        self.rebuild_timer.start()

    def _build_ui(self):
        central = Backdrop()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 20, 28, 18)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.btn_cont = make_button("CONTINUO · t", seg=True)
        self.btn_disc = make_button("DISCRETO · n", seg=True)
        self.btn_cont.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.btn_cont)
        self.mode_group.addButton(self.btn_disc)
        self.btn_cont.clicked.connect(lambda _=False: self._set_mode(False))
        self.btn_disc.clicked.connect(lambda _=False: self._set_mode(True))
        header.addWidget(self.btn_cont)
        header.addWidget(self.btn_disc)
        header.addStretch()
        self.badge = QLabel()
        self.badge.setObjectName("badge")
        header.addWidget(self.badge)
        layout.addLayout(header)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        for builder in (self._build_step1, self._build_step2, self._build_step3,
                        self._build_step4):
            page = QWidget()
            builder(page)
            self.tabs.addTab(page, "")

    @staticmethod
    def _error_label():
        lbl = QLabel()
        lbl.setObjectName("error")
        lbl.setWordWrap(True)
        return lbl

    def _build_step1(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)

        card, lay = make_card("SEÑALES DE ENTRADA")

        self.cap_x = caption("", SOFT)
        lay.addWidget(self.cap_x)
        self.input_x = QLineEdit(PRESETS[0][1])
        self.input_x.textChanged.connect(self._schedule_rebuild)
        lay.addWidget(self.input_x)
        self.err_x = self._error_label()
        lay.addWidget(self.err_x)

        self.cap_h = caption("", SOFT)
        lay.addWidget(self.cap_h)
        self.input_h = QLineEdit(PRESETS[0][2])
        self.input_h.textChanged.connect(self._schedule_rebuild)
        lay.addWidget(self.input_h)
        self.err_h = self._error_label()
        lay.addWidget(self.err_h)

        lay.addWidget(caption("RANGO VISIBLE  (↑ ↓ o rueda)", SOFT))
        self.spin_range = QDoubleSpinBox()
        self.spin_range.setRange(2.0, 30.0)
        self.spin_range.setSingleStep(1.0)
        self.spin_range.setValue(6.0)
        self.spin_range.setPrefix("±  ")
        self.spin_range.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.spin_range.valueChanged.connect(self._schedule_rebuild)
        lay.addWidget(self.spin_range)

        lay.addWidget(caption("EJEMPLOS RÁPIDOS", GREEN_SOFT, spacing=2.2))
        self.chip_grid = QGridLayout()
        self.chip_grid.setSpacing(6)
        lay.addLayout(self.chip_grid)
        self._populate_chips()

        self.help_label = QLabel()
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet(f"color: {MUTED}; font-size: 8pt;")
        lay.addWidget(self.help_label)
        lay.addStretch()
        layout.addWidget(card, 1)

        plot_card, pl = make_card(margins=(10, 10, 10, 10))
        self.fig1, self.canvas1 = _new_figure((6, 5))
        self.ax1 = self.fig1.add_subplot(111)
        pl.addWidget(self.canvas1)
        layout.addWidget(plot_card, 2)

    def _build_step2(self, parent):
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)

        card, lay = make_card("INVERSIÓN ESPEJO")
        self.s2_big = QLabel()
        lay.addWidget(self.s2_big)
        info = QLabel(
            f"<span style='color:{SOFT}'>El programa interpreta tu fórmula y evalúa "
            f"<b style='color:{TEXT}'>la señal reflejada</b> directamente, así que funciona con "
            "cualquier señal, no solo con las de ejemplo.</span>")
        info.setWordWrap(True)
        lay.addWidget(info)
        lay.addWidget(caption("LEYENDA", GREEN_SOFT, spacing=2.2))
        self.s2_dots = [QLabel(), QLabel(), QLabel()]
        for d in self.s2_dots:
            lay.addWidget(d)
        lay.addStretch()
        layout.addWidget(card, 1)

        plot_card, pl = make_card(margins=(10, 10, 10, 10))
        self.fig2, self.canvas2 = _new_figure((6, 5))
        self.ax2 = self.fig2.add_subplot(111)
        pl.addWidget(self.canvas2)
        layout.addWidget(plot_card, 2)

    def _build_step3(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(10)

        # ---- Valor actual de y + tarjeta "la matemática"
        hero = QHBoxLayout()
        hero.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(2)
        self.hero_value = GradientLabel("+0.000")
        left.addWidget(self.hero_value)
        self.hero_caption = caption("", SOFT, size=10, spacing=3.2)
        left.addWidget(self.hero_caption)
        self.hero_sub = QLabel()
        self.hero_sub.setStyleSheet("font-size: 10pt;")
        left.addSpacing(6)
        left.addWidget(self.hero_sub)
        left.addStretch()
        hero.addLayout(left, 3)

        math_card, ml = make_card("LA MATEMÁTICA")
        row1 = QHBoxLayout()
        self.m_big1 = QLabel()
        self.m_big1.setMinimumWidth(130)
        row1.addWidget(self.m_big1)
        r1 = QVBoxLayout()
        r1.setSpacing(1)
        r1.addWidget(caption("TAL CUAL NO SE DESLIZA", SOFT, size=8, spacing=1.2, bold=False))
        r1.addWidget(caption("primero se refleja", RED_SOFT, size=8, spacing=1.0, bold=False))
        row1.addLayout(r1)
        ml.addLayout(row1)

        row2 = QHBoxLayout()
        self.m_big2 = QLabel()
        self.m_big2.setMinimumWidth(130)
        row2.addWidget(self.m_big2)
        r2 = QVBoxLayout()
        r2.setSpacing(1)
        r2.addWidget(caption("REFLEJADA Y DESPLAZADA", SOFT, size=8, spacing=1.2, bold=False))
        self.m_done = caption("", GREEN_SOFT, size=8, spacing=1.0, bold=False)
        r2.addWidget(self.m_done)
        row2.addLayout(r2)
        ml.addLayout(row2)

        self.m_formula = caption("", ORANGE, size=9, spacing=0.8)
        ml.addWidget(self.m_formula)
        hero.addWidget(math_card, 2)
        layout.addLayout(hero)

        # ---- Gráficas
        plot_card, pl = make_card(margins=(10, 10, 10, 10))
        self.fig3, self.canvas3 = _new_figure((10, 6))
        gs = self.fig3.add_gridspec(2, 1, height_ratios=[2.0, 1.4], left=0.055, right=0.985,
                                    top=0.94, bottom=0.09, hspace=0.34)
        self.ax_top = self.fig3.add_subplot(gs[0])
        self.ax_bottom = self.fig3.add_subplot(gs[1])
        pl.addWidget(self.canvas3)
        layout.addWidget(plot_card, 1)

        # ---- Leyenda con puntos de color
        legend = QHBoxLayout()
        legend.setSpacing(22)
        self.s3_dots = [QLabel() for _ in range(5)]
        for d in self.s3_dots:
            legend.addWidget(d)
        legend.addStretch()
        layout.addLayout(legend)

        # ---- Transporte
        transport, tl = make_card(horizontal=True, margins=(16, 12, 16, 12))
        self.btn_play = make_button("▶ REPRODUCIR", kind="primary")
        self.btn_play.clicked.connect(self.toggle_play)
        tl.addWidget(self.btn_play)

        btn_reset = make_button("↺ REINICIAR")
        btn_reset.clicked.connect(self.reset_animation)
        tl.addWidget(btn_reset)

        tl.addSpacing(8)
        tl.addWidget(caption("PROGRESO", MUTED, spacing=2.2))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, N_FRAMES - 1)
        self.slider.sliderPressed.connect(self.pause)
        self.slider.valueChanged.connect(self._on_slider)
        tl.addWidget(self.slider, 1)

        self.lbl_live = QLabel()
        tl.addWidget(self.lbl_live)
        tl.addSpacing(8)

        self.btn_slower = make_button("« LENTO")
        self.btn_slower.clicked.connect(lambda _=False: self.change_speed(-1))
        tl.addWidget(self.btn_slower)

        self.lbl_speed = QLabel()
        self.lbl_speed.setObjectName("speed")
        self.lbl_speed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tl.addWidget(self.lbl_speed)

        self.btn_faster = make_button("RÁPIDO »")
        self.btn_faster.clicked.connect(lambda _=False: self.change_speed(+1))
        tl.addWidget(self.btn_faster)
        layout.addWidget(transport)

    def _build_step4(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(10)

        card, lay = make_card()
        top = QHBoxLayout()
        self.sym_title = caption("", ACCENT_SOFT, spacing=2.2)
        top.addWidget(self.sym_title)
        top.addStretch()
        self.btn_sym = make_button("↻ CALCULAR")
        self.btn_sym.clicked.connect(lambda _=False: self._start_symbolic(force=True))
        top.addWidget(self.btn_sym)
        lay.addLayout(top)

        self.sym_view = QPlainTextEdit()
        self.sym_view.setReadOnly(True)
        self.sym_view.setMinimumHeight(150)
        self.sym_view.setMaximumHeight(230)
        mono = QFont()
        mono.setFamilies(MONO_FAMILIES)
        mono.setPointSize(10)
        self.sym_view.setFont(mono)
        lay.addWidget(self.sym_view)

        row = QHBoxLayout()
        self.eval_caption = caption("", SOFT)
        row.addWidget(self.eval_caption)
        self.eval_t = QLineEdit("1")
        self.eval_t.setFixedWidth(150)
        self.eval_t.textChanged.connect(self._eval_at)
        row.addWidget(self.eval_t)
        self.eval_out = QLabel()
        self.eval_out.setStyleSheet("font-size: 11pt;")
        row.addWidget(self.eval_out)
        row.addStretch()
        lay.addLayout(row)

        self.res_info = QLabel()
        self.res_info.setWordWrap(True)
        self.res_info.setStyleSheet("font-size: 10pt;")
        lay.addWidget(self.res_info)
        layout.addWidget(card)

        plot_card, pl = make_card(margins=(10, 10, 10, 10))
        self.fig4, self.canvas4 = _new_figure((10, 5))
        self.ax4 = self.fig4.add_subplot(111)
        pl.addWidget(self.canvas4)
        layout.addWidget(plot_card, 1)

        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------ modo
    def _populate_chips(self):
        while self.chip_grid.count():
            item = self.chip_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        presets = DISCRETE_PRESETS if self.discrete else PRESETS
        for i, (name, _, _) in enumerate(presets):
            chip = make_button(name, chip=True)
            chip.clicked.connect(lambda _=False, idx=i: self._apply_preset(idx))
            self.chip_grid.addWidget(chip, i // 2, i % 2)

    def _apply_mode_texts(self):
        """Actualiza los textos fijos de la interfaz según el modo elegido."""
        d = self.discrete
        v = "n" if d else "t"
        x, h, y = fn(d, "x", v), fn(d, "h", v), fn(d, "y", v)
        arg_k = "k" if d else "τ"
        h_k, x_k = fn(d, "h", arg_k), fn(d, "x", arg_k)
        h_rev = fn(d, "h", f"−{arg_k}")
        h_shift = fn(d, "h", "n − k" if d else "t − τ")

        self.setWindowTitle(f"Simulador de Convolución — {x} ∗ {h}")
        self.badge.setText(
            f"<span style='color:{GREEN}'>●</span>&nbsp;&nbsp;"
            + ("SUMATORIA · TIEMPO DISCRETO" if d else "INTEGRAL NUMÉRICA · TRAPECIO"))

        for i, title in enumerate((f"1 · SEÑALES {x} Y {h}", f"2 · REFLEXIÓN {h_rev}",
                                   f"3 · CONVOLUCIÓN {y}", f"4 · RESULTADO {y}")):
            self.tabs.setTabText(i, title)

        self.sym_title.setText(f"{y} EXACTA · FÓRMULA ALGEBRAICA")
        self.eval_caption.setText(f"VALOR NUMÉRICO  {y} EN {'n' if d else 't'} =")
        self.cap_x.setText(f"<span style='color:{X_COL}'>●</span> SEÑAL {x}")
        self.cap_h.setText(f"<span style='color:{H_COL}'>●</span> RESPUESTA {h}")
        self.help_label.setText(HELP_DISC if d else HELP_CONT)

        self.s2_big.setText(
            f"<span style='color:{H_COL}; font-size:20pt; font-weight:bold;'>{h_rev}</span>")
        for lbl, (col, txt) in zip(self.s2_dots, (
                (X_COL, f"{x_k} · tenue, referencia"),
                (H_COL, f"{h_k} · punteada, original"),
                (H_COL, f"{h_rev} · sólida, reflejada"))):
            lbl.setText(dot_html(col, txt))

        self.m_big1.setText(
            f"<span style='color:{MUTED}; font-size:20pt; font-weight:bold;'><s>{h_k}</s></span>")
        self.m_big2.setText(
            f"<span style='color:{GREEN}; font-size:20pt; font-weight:bold;'>{h_shift}</span>")
        self.m_done.setText("sumada una vez ✓" if d else "integrada una vez ✓")
        self.m_formula.setText(
            "y[n] = Σ x[k] · h[n − k]" if d else "y(t) = ∫ x(τ) · h(t − τ) dτ")

        for lbl, (col, txt) in zip(self.s3_dots, (
                (X_COL, x_k), (H_COL, h_shift),
                (GREEN, f"{x_k}·{h_shift}  (" + ("la suma es " if d else "el área es ") + y + ")"),
                (TEXT, y), (ORANGE, "instante actual"))):
            lbl.setText(dot_html(col, txt))

    def _set_mode(self, discrete: bool):
        try:
            if discrete == self.discrete:
                return
            self.discrete = discrete
            self.pause()
            self.spin_range.blockSignals(True)
            self.spin_range.setDecimals(0 if discrete else 2)
            if discrete:
                self.spin_range.setValue(float(round(self.spin_range.value())))
            self.spin_range.blockSignals(False)
            self._apply_mode_texts()
            self._populate_chips()
            self.rebuild_timer.stop()
            self._rebuild()
        except Exception:  # noqa: BLE001
            _log(traceback.format_exc())

    # ------------------------------------------------------------ cálculo
    def _apply_preset(self, index):
        presets = DISCRETE_PRESETS if self.discrete else PRESETS
        _, x_expr, h_expr = presets[index]
        for edit, txt in ((self.input_x, x_expr), (self.input_h, h_expr)):
            edit.blockSignals(True)
            edit.setText(txt)
            edit.blockSignals(False)
        self.rebuild_timer.stop()
        self._rebuild()

    def _rebuild(self):
        """Recalcula todo lo que depende de x, h, rango y modo.

        Cualquier excepción se captura aquí: un slot que lanza deja morir la aplicación."""
        try:
            self._rebuild_impl()
        except Exception:  # noqa: BLE001
            _log(traceback.format_exc())
            self.err_x.setText("⚠ Error interno al recalcular (detalle en convolucion_error.log).")

    def _rebuild_impl(self):
        discrete = self.discrete
        R = self.spin_range.value()
        if discrete:
            R = int(round(R))
            tau = np.arange(-2 * R, 2 * R + 1, dtype=float)      # índices k
            t_frames = np.arange(-R, R + 1, dtype=float)         # índices n
            dt = 1.0
        else:
            dt = max(0.01, 4.0 * R / 6000.0)
            n = int(round(4.0 * R / dt)) + 1
            tau = np.linspace(-2.0 * R, 2.0 * R, n)  # cubre el soporte de x y de h(t-τ) en [-R, R]
            dt = float(tau[1] - tau[0])
            t_frames = np.linspace(-R, R, N_FRAMES)

        signals = {}
        for key, edit, label in (("x", self.input_x, self.err_x), ("h", self.input_h, self.err_h)):
            try:
                signals[key] = compile_signal(edit.text(), dt, discrete)
                label.clear()
            except SignalError as exc:
                signals[key] = None
                label.setText(f"⚠ {exc}")
        if signals["x"] is None or signals["h"] is None:
            return  # se conserva la última gráfica válida

        fx, fh = signals["x"], signals["h"]
        try:
            x_vals = fx(tau)
            h_vals = fh(tau)
            h_ref = fh(-tau)
            H = fh(t_frames[:, None] - tau[None, :])   # h(t-τ) / h[n-k] para cada frame
        except SignalError as exc:
            self.err_x.setText(f"⚠ {exc}")
            return
        P = _sane(H * x_vals[None, :])                 # integrando / término de la suma
        y = P.sum(axis=1) if discrete else _TRAPZ(P, tau, axis=1)
        y = _sane(y)

        self.R, self.tau, self.t_frames = R, tau, t_frames
        self.x_vals, self.h_vals, self.h_ref = x_vals, h_vals, h_ref
        self.H, self.P, self.y_frames = H, P, y
        self.n_frames = len(t_frames)
        self.fx, self.fh = fx, fh
        self._d = discrete
        i0 = int(np.searchsorted(tau, -R - 1e-9))
        i1 = int(np.searchsorted(tau, R + 1e-9, side="right"))
        self.sl = slice(i0, i1)  # solo se dibuja la zona visible

        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_frames - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        if discrete:
            self.hero_caption.setText(
                "AMPLITUD DE SALIDA y[n] · "
                f"<span style='color:{GREEN_SOFT}'>n ∈ [−{R:d}, +{R:d}]</span>")
        else:
            self.hero_caption.setText(
                "AMPLITUD DE SALIDA y(t) · "
                f"<span style='color:{GREEN_SOFT}'>t ∈ [−{R:g}, +{R:g}]</span>")

        self._init_step3_artists()
        self._draw_step1()
        self._draw_step2()
        self._draw_step4()
        self._prepare_animation()
        self._sym_key = None                 # la fórmula exacta anterior ya no corresponde
        self._cancel_symbolic()
        self._eval_at()
        if self.tabs.currentIndex() == 3:
            self._start_symbolic()

    # ---------------------------------------------------------- gráficas
    def _xlim(self):
        pad = 0.8 if self._d else 0.0   # margen para que los stems de los extremos no se corten
        return -self.R - pad, self.R + pad

    def _plot_signal(self, ax, xs, ys, color, label, lw=2.5, alpha=1.0, ls="-", hollow=False):
        """Línea en continuo, stem en discreto."""
        if self._d:
            ax.vlines(xs, 0.0, ys, colors=color, linewidths=max(lw * 0.8, 1.2), alpha=alpha,
                      linestyles=ls)
            ax.plot(xs, ys, linestyle="none", marker="o", markersize=5.5, color=color,
                    alpha=alpha, label=label, markerfacecolor="none" if hollow else color)
        else:
            ax.plot(xs, ys, color=color, label=label, linewidth=lw, alpha=alpha, linestyle=ls)

    def _draw_step1(self):
        d, ax, sl, tau = self._d, self.ax1, self.sl, self.tau
        v = "n" if d else "t"
        ax.clear()
        _style_axis(ax, f"PASO 1 · SEÑALES {fn(d, 'x', v)} Y {fn(d, 'h', v)}", integer_x=d)
        self._plot_signal(ax, tau[sl], self.x_vals[sl], X_COL, fn(d, "x", v))
        self._plot_signal(ax, tau[sl], self.h_vals[sl], H_COL, fn(d, "h", v))
        ax.set_xlim(*self._xlim())
        ax.set_ylim(*_limits(self.x_vals[sl], self.h_vals[sl]))
        _legend(ax)
        self.fig1.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.canvas1.draw_idle()

    def _draw_step2(self):
        d, ax, sl, tau = self._d, self.ax2, self.sl, self.tau
        a = "k" if d else "τ"
        ax.clear()
        _style_axis(ax, f"PASO 2 · REFLEXIÓN {fn(d, 'h', '−' + a)}", integer_x=d)
        self._plot_signal(ax, tau[sl], self.x_vals[sl], X_COL, fn(d, "x", a), lw=2.0, alpha=0.4)
        self._plot_signal(ax, tau[sl], self.h_vals[sl], H_COL, fn(d, "h", a), lw=1.8, alpha=0.5,
                          ls="--", hollow=True)
        self._plot_signal(ax, tau[sl], self.h_ref[sl], H_COL, fn(d, "h", "−" + a))
        ax.set_xlim(*self._xlim())
        ax.set_ylim(*_limits(self.x_vals[sl], self.h_vals[sl], self.h_ref[sl]))
        _legend(ax)
        self.fig2.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08)
        self.canvas2.draw_idle()

    def _draw_step4(self):
        """Resultado completo y(t) / y[n] y un resumen numérico."""
        d, ax = self._d, self.ax4
        v = "n" if d else "t"
        t, y = self.t_frames, self.y_frames
        ax.clear()
        _style_axis(ax, f"RESULTADO · {fn(d, 'y', v)} = {fn(d, 'x', v)} ∗ {fn(d, 'h', v)}",
                    integer_x=d)
        if d:
            ax.vlines(t, 0.0, y, colors=TEXT, linewidths=2.0)
            ax.plot(t, y, linestyle="none", marker="o", markersize=6, color=TEXT)
        else:
            ax.plot(t, y, color=TEXT, linewidth=2.5)
        ax.set_xlim(*self._xlim())
        ax.set_ylim(*_limits(y))
        ax.set_xlabel(v, color=ORANGE, fontsize=10, fontweight="bold")
        self.fig4.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.10)
        self.canvas4.draw_idle()

        imax, imin = int(np.argmax(y)), int(np.argmin(y))
        total = float(y.sum()) if d else float(_TRAPZ(y, t))
        total_lbl = "Σ y[n]" if d else "∫ y dt"
        at = (lambda i: f"n = {int(t[i]):+d}") if d else (lambda i: f"t = {t[i]:+.2f}")
        self.res_info.setText(
            f"<span style='color:{SOFT}'>máx = </span><span style='color:{GREEN_SOFT}'>"
            f"{fmt_value(float(y[imax]))}</span><span style='color:{SOFT}'> en {at(imax)}"
            f" &nbsp;·&nbsp; mín = </span><span style='color:{GREEN_SOFT}'>"
            f"{fmt_value(float(y[imin]))}</span><span style='color:{SOFT}'> en {at(imin)}"
            f" &nbsp;·&nbsp; {total_lbl} = </span><span style='color:{ORANGE}'>"
            f"{fmt_value(total)}</span>")

    def _init_step3_artists(self):
        """Crea los artistas de la pestaña 3 (una vez por recálculo, no por frame)."""
        d = self._d
        for ax in (self.ax_top, self.ax_bottom):
            ax.clear()
        self._fill = None
        a = "k" if d else "τ"
        shifted = "n − k" if d else "t − τ"

        _style_axis(self.ax_top, f"TRASLAPE · {fn(d, 'x', a)} · {fn(d, 'h', shifted)}",
                    integer_x=d)
        self.ax_top.set_xlabel(a, color=MUTED, fontsize=9)
        _style_axis(self.ax_bottom, f"RESULTADO · {fn(d, 'y', 'n' if d else 't')}", integer_x=d)
        self.ax_bottom.set_xlabel("n" if d else "t", color=ORANGE, fontsize=10, fontweight="bold")

        if d:
            self.s_x, self.m_x = _make_stem(self.ax_top, X_COL, lw=2.2, ms=6)
            self.s_h, self.m_h = _make_stem(self.ax_top, H_COL, lw=2.2, ms=6)
            self.s_p = LineCollection([], colors=GREEN, linewidths=9, alpha=0.45, zorder=2,
                                      capstyle="butt")
            self.ax_top.add_collection(self.s_p, autolim=False)
            self.s_gh, self.m_gh = _make_stem(self.ax_bottom, TEXT, lw=1.4, alpha=0.16, ms=5)
            self.s_y, self.m_y = _make_stem(self.ax_bottom, TEXT, lw=2.2, ms=6, zorder=4)
        else:
            (self.l_x_top,) = self.ax_top.plot([], [], color=X_COL, lw=2.5)
            (self.l_h_top,) = self.ax_top.plot([], [], color=H_COL, lw=2.5)
            (self.l_ghost,) = self.ax_bottom.plot([], [], color=TEXT, lw=1.5, alpha=0.16)
            (self.l_y,) = self.ax_bottom.plot([], [], color=TEXT, lw=2.5)

        (self.l_pt,) = self.ax_bottom.plot([], [], "o", color=ORANGE, ms=7, zorder=6)
        (self.l_stem,) = self.ax_bottom.plot([], [], color=ORANGE, ls="-" if d else ":",
                                             lw=2.5, zorder=5)

    def _prepare_animation(self):
        sl = self.sl
        if self._d:
            _set_stem(self.s_x, self.m_x, self.tau[sl], self.x_vals[sl])
            _set_stem(self.s_gh, self.m_gh, self.t_frames, self.y_frames)
        else:
            self.l_x_top.set_data(self.tau[sl], self.x_vals[sl])
            self.l_ghost.set_data(self.t_frames, self.y_frames)  # y(t) completa, tenue, de guía

        self.ax_top.set_xlim(*self._xlim())
        self.ax_top.set_ylim(*_limits(self.x_vals[sl], self.h_vals[sl], self.H[:, sl],
                                      self.P[:, sl]))
        self.ax_bottom.set_xlim(*self._xlim())
        self.ax_bottom.set_ylim(*_limits(self.y_frames))

        self.pause()
        self._pos = 0.0
        self._shown = -1
        self._show_frame(0, force=True)

    # ---------------------------------------------------------- animación
    def _show_frame(self, idx, force=False):
        if self.y_frames is None:
            return
        idx = int(np.clip(idx, 0, self.n_frames - 1))
        if idx == self._shown and not force:
            return  # a velocidades bajas varios ticks caen en el mismo frame
        self._shown = idx

        sl = self.sl
        t_val = float(self.t_frames[idx])
        y_cur = float(self.y_frames[idx])
        xs = self.tau[sl]

        if self._d:
            _set_stem(self.s_h, self.m_h, xs, self.H[idx, sl])
            self.s_p.set_segments(_segs(xs, self.P[idx, sl]))
            _set_stem(self.s_y, self.m_y, self.t_frames[:idx + 1], self.y_frames[:idx + 1])
            self.hero_sub.setText(
                f"en <span style='color:{GREEN_SOFT}; font-weight:bold;'>n = {int(round(t_val)):+d}"
                "</span> — suma de x[k]·h[n − k]")
        else:
            self.l_h_top.set_data(xs, self.H[idx, sl])
            if self._fill is not None:
                try:
                    self._fill.remove()
                except (ValueError, NotImplementedError):
                    pass
            self._fill = self.ax_top.fill_between(xs, 0.0, self.P[idx, sl],
                                                  facecolor=GREEN, alpha=0.38, linewidth=0)
            self.l_y.set_data(self.t_frames[:idx + 1], self.y_frames[:idx + 1])
            self.hero_sub.setText(
                f"en <span style='color:{GREEN_SOFT}; font-weight:bold;'>t = {t_val:+.2f}</span>"
                " — área bajo x(τ)·h(t − τ)")

        self.l_pt.set_data([t_val], [y_cur])
        self.l_stem.set_data([t_val, t_val], [0.0, y_cur])
        self.hero_value.setText(fmt_value(y_cur))

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        self.canvas3.draw_idle()

    # ------------------------------------------------- fórmula exacta (SymPy)
    def _on_tab_changed(self, index):
        if index == 3:
            self._start_symbolic()

    def _start_symbolic(self, force=False):
        """Lanza `python main.py --symbolic ...` en un proceso aparte (se puede cortar)."""
        try:
            d = self.discrete
            key = (d, self.input_x.text(), self.input_h.text(), self.R)
            if not force and self._sym_key == key:
                return
            if self.x_vals is None or self._d != d:
                return                      # las entradas actuales no son válidas todavía
            self._cancel_symbolic()
            self._sym_key = key
            self.sym_view.setPlainText("Calculando la fórmula exacta con SymPy…")
            proc = QProcess(self)
            proc.setProgram(sys.executable)
            proc.setArguments([os.path.abspath(__file__), "--symbolic", "disc" if d else "cont",
                               self.input_x.text(), self.input_h.text(), repr(float(self.R))])
            proc.finished.connect(self._on_symbolic_finished)
            proc.errorOccurred.connect(self._on_symbolic_error)
            self._sym_proc = proc
            proc.start()
            self.sym_timeout.start()
        except Exception:  # noqa: BLE001
            _log(traceback.format_exc())
            self.sym_view.setPlainText("No se pudo iniciar el cálculo exacto (detalle en convolucion_error.log).")

    def _cancel_symbolic(self):
        self.sym_timeout.stop()
        proc, self._sym_proc = self._sym_proc, None
        if proc is not None:
            try:
                proc.finished.disconnect()
                proc.errorOccurred.disconnect()
            except (TypeError, RuntimeError):
                pass
            proc.kill()
            proc.deleteLater()

    def _on_symbolic_finished(self, *_args):
        try:
            proc = self._sym_proc
            if proc is None:
                return
            self.sym_timeout.stop()
            raw = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="replace").strip()
            err = bytes(proc.readAllStandardError()).decode("utf-8", errors="replace").strip()
            self._sym_proc = None
            proc.deleteLater()
            try:
                res = json.loads(raw)
            except ValueError:
                msg = "No se pudo calcular la fórmula exacta."
                if "No module named 'sympy'" in err:
                    msg += "\nFalta SymPy:  pip install sympy"
                elif err:
                    _log(err)
                res = {"ok": False, "reason": msg}
            self._show_symbolic_result(res)
        except Exception:  # noqa: BLE001
            _log(traceback.format_exc())

    def _on_symbolic_error(self, *_args):
        if self._sym_proc is not None and self._sym_proc.state() == QProcess.ProcessState.NotRunning:
            self._on_symbolic_finished()

    def _on_symbolic_timeout(self):
        self._cancel_symbolic()
        self.sym_view.setPlainText(
            "El cálculo exacto tardó más de 60 s y se canceló.\n"
            "Para señales complicadas usa el valor numérico de abajo.")

    def _show_symbolic_result(self, res):
        if res.get("ok"):
            text = res["text"] + "\n\n(comprobada contra el cálculo numérico)"
        else:
            text = ("No se pudo obtener una fórmula cerrada para estas señales.\n"
                    f"Motivo: {res.get('reason', 'desconocido')}\n\n"
                    "Usa el valor numérico de abajo o la gráfica.")
        self.sym_view.setPlainText(text)

    def _eval_at(self, *_args):
        """Valor numérico de y en el instante escrito (acepta 1.5, pi/2, -3...)."""
        try:
            if self.fx is None:
                return
            d = self._d
            raw = self.eval_t.text().strip().replace(",", ".")
            if not raw:
                self.eval_out.setText("")
                return
            t0 = float(compile_signal(raw, 1.0, d)(np.array([0.0]))[0])
            R = self.R
            half = 2 * R + abs(t0)
            if d:
                t0 = float(round(t0))
                half = int(round(half))
                k = np.arange(-half, half + 1, dtype=float)
                y = float(np.sum(self.fx(k) * self.fh(t0 - k)))
                shown = f"n = {int(t0):+d}"
            else:
                dt = 0.001      # paso fino: un solo punto, así que sale barato
                k = np.arange(-half, half + dt / 2, dt)
                dtf = float(k[1] - k[0])
                fx, fh = compile_signal(self.input_x.text(), dtf, False), \
                    compile_signal(self.input_h.text(), dtf, False)
                y = float(_TRAPZ(fx(k) * fh(t0 - k), k))
                shown = f"t = {t0:+.4g}"
            y = float(_sane(y))
            self.eval_out.setText(
                f"<span style='color:{SOFT}'>{shown} &nbsp;→&nbsp; {fn(d, 'y', 'n' if d else 't')} {'=' if d else '≈'} "
                f"</span><span style='color:{GREEN_SOFT}; font-weight:bold;'>{fmt_value(y)}</span>")
        except SignalError as exc:
            self.eval_out.setText(f"<span style='color:{RED_SOFT}'>⚠ {exc}</span>")
        except Exception:  # noqa: BLE001
            _log(traceback.format_exc())

    def _step_size(self):
        base = DISCRETE_STEP if self._d else 1.0
        return base * SPEEDS[self.speed_idx]

    def _tick(self):
        try:
            self._pos = min(self._pos + self._step_size(), self.n_frames - 1)
            self._show_frame(int(self._pos))
            if self._pos >= self.n_frames - 1:
                self.pause()
        except Exception:  # noqa: BLE001
            _log(traceback.format_exc())
            self.pause()  # mejor detener la animación que perder la ventana

    def play(self):
        if self._pos >= self.n_frames - 1:
            self._pos = 0.0
        self.is_playing = True
        self.timer.start()
        self._update_live_ui()

    def pause(self):
        self.timer.stop()
        self.is_playing = False
        self._update_live_ui()

    def toggle_play(self):
        self.pause() if self.is_playing else self.play()

    def reset_animation(self):
        self.pause()
        self._pos = 0.0
        self._show_frame(0, force=True)

    def _on_slider(self, value):
        self.pause()
        self._pos = float(value)
        self._show_frame(value)

    def change_speed(self, step):
        self.speed_idx = int(np.clip(self.speed_idx + step, 0, len(SPEEDS) - 1))
        self._update_speed_ui()

    def _update_speed_ui(self):
        self.lbl_speed.setText(f"VELOCIDAD {SPEEDS[self.speed_idx]:g}×")
        self.btn_slower.setEnabled(self.speed_idx > 0)
        self.btn_faster.setEnabled(self.speed_idx < len(SPEEDS) - 1)

    def _update_live_ui(self):
        if self.is_playing:
            self.btn_play.setText("❚❚ PAUSA")
            self.lbl_live.setText(f"<span style='color:{GREEN}'>●</span>"
                                  f"&nbsp;<span style='color:{GREEN_SOFT}'>EN VIVO</span>")
        else:
            self.btn_play.setText("▶ REPRODUCIR")
            self.lbl_live.setText(f"<span style='color:{MUTED}'>●&nbsp;EN PAUSA</span>")

    def closeEvent(self, event):
        self.timer.stop()
        self.rebuild_timer.stop()
        self._cancel_symbolic()
        super().closeEvent(event)


def main():
    global _FAULT_LOG
    if len(sys.argv) > 1 and sys.argv[1] == "--symbolic":   # modo interno: fórmula exacta
        _symbolic_cli(sys.argv[2:])
        return
    sys.excepthook = _excepthook  # clave: sin esto PyQt6 llama a abort() ante cualquier excepción
    try:
        _FAULT_LOG = open(_LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_LOG)  # deja rastro incluso si el cierre es nativo
    except Exception:  # noqa: BLE001
        pass
    app = QApplication(sys.argv)
    window = ConvolutionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
