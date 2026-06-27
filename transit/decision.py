"""
transit.decision — re-export of the adaptive decision module.

The decision engine's canonical implementation lives in the top-level
``decision`` package (dissertation section 3.2), which the test-suite and the
README already reference as ``from decision import DecisionEngine``. This module
re-exports it under the ``transit`` namespace so the library has one tidy public
surface (``from transit import DecisionEngine``) without duplicating code.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

# Make the repo root importable so the sibling ``decision`` package is found
# regardless of the current working directory.
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from decision import (  # noqa: E402,F401
    Action,
    ACTIONS,
    Decision,
    DecisionEngine,
    IncrementalCPDUpdater,
)

__all__ = [
    "Action",
    "ACTIONS",
    "Decision",
    "DecisionEngine",
    "IncrementalCPDUpdater",
]
