"""計算エンジン群。

標準で 3 種類を同梱している:

``qe``
    Quantum ESPRESSO (``pw.x``、``dos.x``、``projwfc.x``)。
``vasp``
    VASP、または aiida-vasp の ``mock-vasp``。後者は記録済みの計算を再生するため、
    ライセンスが無い環境でも VASP 経路を動かせる。
``mlip``
    ASE 経由の機械学習ポテンシャル (既定は SevenNet)。

新しいコードを追加する場合は :class:`ezcal.engines.base.Engine` を継承した
モジュールを 1 つ書き、下記に登録するだけでよい。CLI・設定層・ワークフローは
変更する必要がない。
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
