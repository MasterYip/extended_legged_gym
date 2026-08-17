"""Compatibility facade for :mod:`el4090_envelope.model`.

New code should import ``el4090_envelope``. This module deliberately contains
no model implementation; it exposes the package module's complete historical
namespace, including private helpers used by legacy tests.
"""

from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[4] / "el4090_envelope" / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from el4090_envelope import model as _model  # noqa: E402

globals().update({
    name: getattr(_model, name)
    for name in dir(_model)
    if not name.startswith("__")
})

__all__ = [name for name in dir(_model) if not name.startswith("_")]
