"""
Shared pytest helpers.

Two things the test-suite needs:
  * the repo root on sys.path, so ``import config`` and ``import decision`` work;
  * a loader for the digit-prefixed pipeline scripts (``04_run_experiment.py``),
    whose names are not valid Python identifiers and therefore cannot be
    imported with a plain ``import``.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_script(filename: str):
    """Import a pipeline script by file name and return the module object."""
    path = ROOT / filename
    module_name = "pipeline_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
