"""Test-time sys.path setup for the v2 pipeline.

This file lets pytest import both the v2 package and v1 modules
(``pipeline_core.storage`` etc.) without requiring ``pip install -e .``
against the shared venv. Two directories are added to ``sys.path``:

  - This directory (``pipeline_v2/``) so ``from pipeline_v2 import models``
    resolves to the inner ``pipeline_v2/pipeline_v2/`` regular package.
  - The parent ``KaizerBackend/`` so v1 modules like
    ``pipeline_core.storage`` resolve.

Regular packages (with ``__init__.py``) take precedence over namespace
packages per PEP 420, so the inner ``pipeline_v2`` package wins over the
outer ``pipeline_v2/`` namespace directory even when both are reachable.

Once we set up an editable install in CI we can delete this file.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent              # .../pipeline_v2/
_KAIZER_BACKEND = _HERE.parent                       # .../KaizerBackend/

for _p in (_HERE, _KAIZER_BACKEND):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
