"""AST checker for Python 3.6 compatibility.

Complements ``vermin``.  vermin catches most things but, as of 1.8, it does
*not* flag PEP 604 unions (``int | None``) inside annotations nor
``from __future__ import annotations``, both of which are hard failures on
3.6.  This checker covers those and the rest of the constructs the package is
required to avoid.

Run standalone::

    python tests/py36_ast_check.py scorecard_segment_eval tests
"""

import ast
import os
import sys

#: Attribute / call names that only exist from 3.7 onwards.
FORBIDDEN_ATTRIBUTES = {
    "cached_property": "functools.cached_property requires 3.8",
    "prod": "math.prod requires 3.8 (only flagged on the math module)",
    "removeprefix": "str.removeprefix requires 3.9",
    "removesuffix": "str.removesuffix requires 3.9",
    "nullcontext": "contextlib.nullcontext requires 3.7",
    "get_origin": "typing.get_origin requires 3.8",
    "get_args": "typing.get_args requires 3.8",
    "final": "typing.final requires 3.8",
}

#: Modules that do not exist on 3.6.
FORBIDDEN_MODULES = {
    "dataclasses": None,  # allowed: declared as a conditional dependency
    "zoneinfo": "zoneinfo requires 3.9",
    "graphlib": "graphlib requires 3.9",
    "importlib.resources": "importlib.resources requires 3.7",
    "contextvars": "contextvars requires 3.7",
}

#: typing names that only exist from 3.8 onwards.
FORBIDDEN_TYPING_NAMES = {
    "Protocol": "typing.Protocol requires 3.8",
    "TypedDict": "typing.TypedDict requires 3.8",
    "Literal": "typing.Literal requires 3.8",
    "Final": "typing.Final requires 3.8",
}

#: Lower-cased builtin containers that cannot be subscripted before 3.9.
PEP585_NAMES = ("list", "dict", "set", "frozenset", "tuple", "type")


class _Visitor(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.problems = []
        self._typing_aliases = set()

    def _add(self, node, message):
        self.problems.append(
            "%s:%d: %s" % (self.path, getattr(node, "lineno", 0), message)
        )

    # -- syntax level ----------------------------------------------------
    def visit_NamedExpr(self, node):  # pragma: no cover - 3.8+ parsers only
        self._add(node, "walrus operator (:=) requires 3.8")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    self._add(
                        node, "'from __future__ import annotations' requires 3.7"
                    )
        if node.module == "typing":
            for alias in node.names:
                if alias.name in FORBIDDEN_TYPING_NAMES:
                    self._add(node, FORBIDDEN_TYPING_NAMES[alias.name])
                self._typing_aliases.add(alias.asname or alias.name)
        if node.module and node.module in FORBIDDEN_MODULES:
            reason = FORBIDDEN_MODULES[node.module]
            if reason:
                self._add(node, reason)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            reason = FORBIDDEN_MODULES.get(alias.name)
            if reason:
                self._add(node, reason)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        reason = FORBIDDEN_ATTRIBUTES.get(node.attr)
        if reason:
            base = getattr(node.value, "id", None)
            if node.attr == "prod" and base != "math":
                pass
            elif node.attr in ("removeprefix", "removesuffix"):
                self._add(node, reason)
            elif base in ("functools", "math", "contextlib", "typing"):
                self._add(node, reason)
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        # f-string '=' specifier (3.8) shows up as a literal '=' immediately
        # before a FormattedValue that carries the original source text.
        for i, value in enumerate(node.values):
            if not isinstance(value, ast.FormattedValue):
                continue
            previous = node.values[i - 1] if i else None
            if (
                isinstance(previous, ast.Constant)
                and isinstance(previous.value, str)
                and previous.value.endswith("=")
                and not previous.value.endswith("==")
            ):
                self._add(node, "f-string '=' specifier requires 3.8")
        self.generic_visit(node)

    # -- annotations -----------------------------------------------------
    def _check_annotation(self, node):
        if node is None:
            return
        for child in ast.walk(node):
            if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
                self._add(child, "PEP 604 union (X | Y) in an annotation requires 3.10")
            if isinstance(child, ast.Subscript):
                base = child.value
                if isinstance(base, ast.Name) and base.id in PEP585_NAMES:
                    self._add(
                        child,
                        "PEP 585 builtin generic (%s[...]) requires 3.9" % (base.id,),
                    )

    def visit_AnnAssign(self, node):
        self._check_annotation(node.annotation)
        self.generic_visit(node)

    def _check_args(self, node):
        args = node.args
        if getattr(args, "posonlyargs", None):
            self._add(node, "positional-only parameters (/) require 3.8")
        every = list(args.args) + list(args.kwonlyargs)
        if args.vararg:
            every.append(args.vararg)
        if args.kwarg:
            every.append(args.kwarg)
        for arg in every:
            self._check_annotation(getattr(arg, "annotation", None))
        self._check_annotation(node.returns)

    def visit_FunctionDef(self, node):
        self._check_args(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._check_args(node)
        self.generic_visit(node)


def check_file(path):
    """Return a list of 3.7+ constructs found in ``path``."""
    with open(path, "r") as handle:
        source = handle.read()
    tree = ast.parse(source, filename=path)
    visitor = _Visitor(path)
    visitor.visit(tree)
    return visitor.problems


def iter_python_files(roots):
    for root in roots:
        if os.path.isfile(root) and root.endswith(".py"):
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in (".git", "__pycache__", ".venv", ".pytest_cache")
            ]
            for name in sorted(filenames):
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def check_paths(roots):
    """Return every violation found under ``roots``."""
    problems = []
    for path in sorted(iter_python_files(roots)):
        problems.extend(check_file(path))
    return problems


def main(argv):
    roots = argv[1:] or ["scorecard_segment_eval", "tests"]
    problems = check_paths(roots)
    scanned = len(list(iter_python_files(roots)))
    if problems:
        for problem in problems:
            print(problem)
        print("FAIL: %d Python 3.7+ construct(s) in %d file(s)" % (len(problems), scanned))
        return 1
    print("OK: %d file(s) scanned, no Python 3.7+ constructs found" % (scanned,))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
