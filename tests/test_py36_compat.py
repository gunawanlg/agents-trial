import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from py36_ast_check import check_file, check_paths  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOTS = [
    os.path.join(REPO_ROOT, "scorecard_segment_eval"),
    os.path.join(REPO_ROOT, "tests"),
]


def test_package_and_tests_are_py36_clean():
    problems = check_paths(ROOTS)
    assert problems == [], "Python 3.7+ constructs found:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    "source,needle",
    [
        ("from __future__ import annotations\n", "__future__ import annotations"),
        ("def f(x: int | None) -> None:\n    return None\n", "PEP 604"),
        ("def f(x: list[int]) -> None:\n    return None\n", "PEP 585"),
        ("import functools\n\n\nclass A:\n    @functools.cached_property\n    def b(self):\n        return 1\n", "cached_property"),
        ("x: dict[str, int] = {}\n", "PEP 585"),
    ],
)
def test_checker_catches_known_violations(tmp_path, source, needle):
    """The checker has to fail on things vermin misses, or it proves nothing."""
    path = str(tmp_path / "sample.py")
    with open(path, "w") as handle:
        handle.write(source)
    problems = check_file(path)
    assert problems, "checker did not flag: " + source
    assert any(needle in problem for problem in problems), problems


def test_checker_accepts_py36_equivalents(tmp_path):
    source = (
        "from typing import Dict, List, Optional\n"
        "\n"
        "\n"
        "def f(x, y=None):\n"
        "    # type: (List[int], Optional[Dict[str, int]]) -> str\n"
        "    return '%d' % (len(x),)\n"
    )
    path = str(tmp_path / "ok.py")
    with open(path, "w") as handle:
        handle.write(source)
    assert check_file(path) == []


def test_vermin_reports_at_most_36():
    """Mechanical confirmation from vermin when it is installed."""
    vermin = os.path.join(os.path.dirname(sys.executable), "vermin")
    if not os.path.exists(vermin):
        pytest.skip("vermin is not installed")
    proc = subprocess.Popen(
        [
            vermin,
            "-t=3.6-",
            "--violations",
            "--eval-annotations",
            "--backport",
            "dataclasses",
            "--backport",
            "typing",
            "--no-tips",
            "scorecard_segment_eval",
            "tests",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    out = proc.communicate()[0].decode("utf-8", "replace")
    assert proc.returncode == 0, out
    assert "Minimum required versions: 3.6" in out, out
    assert "Target versions not met" not in out, out
