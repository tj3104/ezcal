"""Calculation engines.

Three are shipped:

``qe``
    Quantum ESPRESSO (``pw.x``, ``dos.x``, ``projwfc.x``).
``vasp``
    VASP, or aiida-vasp's ``mock-vasp`` which replays recorded runs so the
    VASP path works without a licence.
``mlip``
    A machine-learning potential through ASE (SevenNet by default).

Adding a new code means writing one module that subclasses
:class:`ezcal.engines.base.Engine` and registering it below - the CLI, the
configuration layer and the workflows do not change.
"""

from __future__ import annotations

from typing import Callable

from ezcal.engines.base import CalcResult, Engine, EngineError

_REGISTRY: dict[str, Callable[..., Engine]] = {}


def register(name: str, factory: Callable[..., Engine]) -> None:
    _REGISTRY[name.lower()] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def get_engine(name: str, config, scheduler=None) -> Engine:
    key = str(name).lower()
    if key not in _REGISTRY:
        raise EngineError(f"unknown engine {name!r}; available: {', '.join(available())}")
    return _REGISTRY[key](config, scheduler)


def _qe(config, scheduler):
    from ezcal.engines.qe.engine import QEEngine

    return QEEngine(config, scheduler)


def _vasp(config, scheduler):
    from ezcal.engines.vasp import VaspEngine

    return VaspEngine(config, scheduler)


def _mlip(config, scheduler):
    from ezcal.engines.mlip import MLIPEngine

    return MLIPEngine(config, scheduler)


register("qe", _qe)
register("espresso", _qe)
register("vasp", _vasp)
register("mlip", _mlip)
register("sevennet", _mlip)

__all__ = ["CalcResult", "Engine", "EngineError", "get_engine", "register", "available"]
