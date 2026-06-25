from __future__ import annotations

from pathlib import Path
from typing import Any


def infer_cwru_label(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "normal" in name:
        return "healthy"
    if name.startswith(("ir", "b", "or")) or any(token in name for token in ("_ir", "_b", "_or")):
        return "faulty"
    raise ValueError(f"cannot infer CWRU label from {path!s}")


def extract_drive_end_signal(mat: dict[str, Any]) -> tuple[float, ...]:
    de_keys = sorted(key for key in mat if "DE_time" in key)
    if not de_keys:
        raise ValueError("no drive-end accelerometer channel matching '*DE_time' found")
    return _flatten_numeric(mat[de_keys[0]])


def load_cwru_mat(path: str | Path) -> tuple[tuple[float, ...], str]:
    from scipy.io import loadmat

    mat = loadmat(path)
    return extract_drive_end_signal(mat), infer_cwru_label(path)


def _flatten_numeric(value: Any) -> tuple[float, ...]:
    try:
        import numpy as np

        return tuple(float(item) for item in np.asarray(value).ravel())
    except Exception:
        flattened: list[float] = []
        _flatten_into(value, flattened)
        return tuple(flattened)


def _flatten_into(value: Any, output: list[float]) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_into(item, output)
    else:
        output.append(float(value))
