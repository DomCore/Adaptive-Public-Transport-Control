"""
Backwards-compatible configuration shim.

The canonical configuration now lives in ``transit/config.py`` so that the
library, the ``transit-control`` CLI and the Streamlit dashboard share a single
source of truth. This module simply re-exports everything from there, which
keeps the numbered pipeline scripts (``01_prepare_data.py`` …) and the test
suite working unchanged via ``from config import ...`` / ``import config``.
"""

# Ensure the repository root (this file's directory) is importable so that the
# ``transit`` package can be found even when a numbered script is launched from
# elsewhere. Normally the script's own directory is already on sys.path; this is
# a defensive belt-and-braces guard for editable installs and odd launchers.
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from transit.config import *  # noqa: F401,F403  (re-export the canonical config)
