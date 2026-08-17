"""Compatibility facade for :mod:`el4090_envelope.geometry`."""

from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[4] / "el4090_envelope" / "src"
if _SOURCE_ROOT.is_dir() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from el4090_envelope import geometry as _geometry  # noqa: E402

globals().update({
    name: getattr(_geometry, name)
    for name in dir(_geometry)
    if not name.startswith("__")
})

__all__ = [name for name in dir(_geometry) if not name.startswith("_")]
