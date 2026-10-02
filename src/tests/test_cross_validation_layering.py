"""Structural guards for the cross_validation package.

The package was split out of a single 7,555-line module whose three-way import
cycle (cross_val <-> training_outputs <-> domain_generalization) was
worked around with imports deferred into function bodies. These tests assert the
layering that made those workarounds unnecessary, so the cycle cannot creep back.
"""

import ast
import warnings
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "eegproc" / "deep_learning" / "cross_validation"

# Every import must point strictly downward.
LEVELS = {
    "constants": 0, "arrays": 0, "workers": 0,
    "probabilities": 1, "splits": 1,
    "aggregation": 2, "metrics": 2,
    "callbacks": 3, "reporting": 3,
    "search": 4, "evaluation": 4,
    "loso_fold": 5, "calibration_subject": 5, "nested_fold": 5,
    "loso": 6, "calibration": 6, "nested": 6,
}


def _sibling_imports(path: Path):
    """Yield (importing_module, imported_sibling) for same-package imports."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            yield path.stem, node.module.split(".")[0]


def test_every_module_has_a_declared_level():
    on_disk = {p.stem for p in PACKAGE.glob("*.py")} - {"__init__", "dataframe"}
    assert on_disk == set(LEVELS), (
        "a module was added or renamed without being placed in the layering"
    )


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.stem)
def test_imports_point_strictly_downward(path):
    if path.stem in {"__init__", "dataframe"}:
        pytest.skip("public surface may import any layer")
    for importer, imported in _sibling_imports(path):
        if imported not in LEVELS:
            continue
        assert LEVELS[imported] < LEVELS[importer], (
            f"{importer} (L{LEVELS[importer]}) imports {imported} "
            f"(L{LEVELS[imported]}) — this reintroduces a cycle"
        )


def test_arrays_layer_imports_without_tensorflow():
    """The lowest layer must import with TensorFlow unavailable.

    ``arrays`` is what ``training_outputs`` and the domain-generalization strategy
    modules sit on, and it annotates ``model: tf.keras.Model``. That annotation is
    lazy (``from __future__ import annotations``) and the import is behind
    ``TYPE_CHECKING``, so no TensorFlow import should happen at runtime. Proven in
    a subprocess where importing tensorflow raises.
    """
    import subprocess
    import sys

    script = (
        "import importlib.util, sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'tensorflow' or name.startswith('tensorflow.'):\n"
        "            raise ImportError('tensorflow is blocked for this test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        # load arrays.py standalone: the package __init__ eagerly pulls the whole
        # stack, so importing it normally would prove nothing about this module.
        "spec = importlib.util.spec_from_file_location('_arrays', sys.argv[1])\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert 'tensorflow' not in sys.modules\n"
        "assert m._as_numpy_1d is not None\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(PACKAGE / "arrays.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "arrays.py pulled in TensorFlow at import time:\n" + proc.stderr[-2000:]
    )
    assert "ok" in proc.stdout


def test_no_imports_deferred_into_function_bodies():
    """Deferred imports were the symptom of the cycle; none should remain."""
    offenders = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module and (
                    sub.level > 0 or sub.module.startswith("eegproc")
                ):
                    offenders.append(f"{path.stem}.{node.name} -> {'.' * sub.level}{sub.module}")
    assert not offenders, "intra-package imports deferred into function bodies: " + "; ".join(offenders)


def test_deprecated_shim_still_re_exports_the_public_surface():
    # These two actually import the package; the rest are pure AST checks that
    # work on a base install.
    pytest.importorskip("tensorflow", reason="requires eegproc[deep-learning]")
    pytest.importorskip("sklearn", reason="requires eegproc[deep-learning]")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import importlib

        import eegproc.deep_learning.cross_val as shim

        importlib.reload(shim)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "importing the old module path must warn"
    )
    for name in ("loso_cv", "fixed_loso_cv", "subject_calibration_cv", "nested_lnso_cv"):
        assert hasattr(shim, name), f"shim dropped {name}"


def test_shim_does_not_re_export_private_helpers():
    # These two actually import the package; the rest are pure AST checks that
    # work on a base install.
    pytest.importorskip("tensorflow", reason="requires eegproc[deep-learning]")
    pytest.importorskip("sklearn", reason="requires eegproc[deep-learning]")
    import eegproc.deep_learning.cross_val as shim

    for private in ("_as_numpy_1d", "_run_loso_fold", "_classification_metrics"):
        assert not hasattr(shim, private), f"{private} leaked through the shim"
