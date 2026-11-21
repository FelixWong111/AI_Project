"""Enumerate friendly names using the same backend as OpenCV capture."""
from __future__ import annotations

import cv2

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}


def list_cameras(backend: str = "auto") -> list[dict]:
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as exc:
        raise RuntimeError("Camera names require: python -m pip install -r requirements-p07.txt") from exc
    choices = list(BACKENDS) if backend in ("auto", "any") else [backend]
    return [dict(index=device.index, name=device.name, backend=name)
            for name in choices for device in enumerate_cameras(BACKENDS[name])]


def select_camera(name: str, backend: str = "auto", devices: list[dict] | None = None) -> dict:
    devices = list_cameras(backend) if devices is None else devices
    choices = list(BACKENDS) if backend in ("auto", "any") else [backend]
    for choice in choices:
        candidates = [d for d in devices if d["backend"] == choice and name.casefold() in d["name"].casefold()]
        exact = [d for d in candidates if d["name"].casefold() == name.casefold()]
        candidates = exact or candidates
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError("Ambiguous camera name; use --camera INDEX --capture-backend " + choice)
    raise ValueError("Camera name not found: " + name + ". Run app.py --list-cameras")
