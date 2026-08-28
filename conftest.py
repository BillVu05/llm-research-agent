"""Make the agent and eval modules importable in a fresh clone.

Replaces the sys.path.insert that used to sit at the top of every test module.
The project is also pip-installable (see [project] in pyproject.toml); this
exists so `pytest` works with nothing installed at all.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for _sub in ("src", "eval"):
    sys.path.insert(0, str(_ROOT / _sub))
