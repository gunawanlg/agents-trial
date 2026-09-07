"""Guard Python 3.6 syntax: no walrus, match, PEP585 builtins, future annotations, `|` unions."""

import ast
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..", "scorecard_segment_eval")


def _py_files():
    files = []
    for dirpath, _dirnames, filenames in os.walk(os.path.abspath(ROOT)):
        for name in filenames:
            if name.endswith(".py"):
                files.append(os.path.join(dirpath, name))
    return files


class _Py36Ban(ast.NodeVisitor):
    def __init__(self, path):
        self.path = path
        self.errors = []

    def visit_ImportFrom(self, node):
        if node.module == "__future__" and any(a.name == "annotations" for a in node.names):
            self.errors.append("%s:%s: from __future__ import annotations" % (self.path, node.lineno))
        self.generic_visit(node)

    def visit_NamedExpr(self, node):  # walrus, 3.8+
        self.errors.append("%s:%s: walrus operator" % (self.path, node.lineno))
        self.generic_visit(node)

    def visit_Match(self, node):  # 3.10+
        self.errors.append("%s:%s: match/case" % (self.path, node.lineno))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._check_annotation(node.annotation, node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.returns is not None:
            self._check_annotation(node.returns, node.lineno)
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.annotation is not None:
                self._check_annotation(arg.annotation, node.lineno)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def _check_annotation(self, node, lineno):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            self.errors.append("%s:%s: PEP604 union with |" % (self.path, lineno))
        if isinstance(node, ast.Subscript):
            slc = node.value
            name = None
            if isinstance(slc, ast.Name):
                name = slc.id
            if name in ("list", "dict", "tuple", "set", "type"):
                self.errors.append("%s:%s: builtin generic %s[]" % (self.path, lineno, name))


def test_package_parses_without_post36_syntax():
    errors = []
    for path in _py_files():
        with open(path, "r") as handle:
            src = handle.read()
        if "from __future__ import annotations" in src:
            errors.append(path + ": future annotations")
        if re.search(r":=", src):
            # comments might have := ; AST is the source of truth
            pass
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError as exc:
            errors.append("%s: SyntaxError %s" % (path, exc))
            continue
        visitor = _Py36Ban(path)
        visitor.visit(tree)
        errors.extend(visitor.errors)
    assert errors == []
