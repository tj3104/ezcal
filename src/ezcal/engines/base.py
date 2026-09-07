"""すべての計算バックエンドが実装するエンジンインターフェース。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: ezcal 全体で共通して使うタスク名
TASKS = ("scf", "relax", "vc-relax", "nscf", "bands", "dos", "pdos")


class EngineError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class CalcResult:
    """1 ステップ分の計算結果 (エンジン非依存の共通表現)。"""

    task: str
    engine: str
    ok: bool = False
    workdir: Path = field(default_factory=Path)
    prefix: str = "ezcal"

    energy: float | None = None                 # eV、全エネルギー
    energy_per_atom: float | None = None        # eV/原子
    fermi_energy: float | None = None           # eV
    homo: float | None = None
    lumo: float | None = None
    band_gap: float | None = None               # eV
    magnetization: float | None = None          # ボーア磁子/セル (正味の値)
    abs_magnetization: float | None = None      # |m(r)| の積分。反強磁性でも 0 にならない
    site_magnetization: list[float] | None = None   # 原子ごと、ボーア磁子
    forces: list[list[float]] | None = None     # eV/Angstrom
    max_force: float | None = None
    stress: list[list[float]] | None = None     # GPa
    pressure: float | None = None               # GPa
    structure: Any = None                       # pymatgen の Structure (最適化後)
    nelec: float | None = None
    nbnd: int | None = None
    kmesh: list[int] | None = None
    nkpt: int | None = None
    converged: bool = False
    walltime: float | None = None
    files: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)   # バンド / DOS のデータ本体
    messages: list[str] = field(default_factory=list)
    job_id: str | None = None
    submitted_only: bool = False

    # -- シリアライズ ------------------------------------------------------
    def summary(self) -> dict:
        keys = (
            "task", "engine", "ok", "converged", "energy", "energy_per_atom",
            "fermi_energy", "band_gap", "homo", "lumo", "magnetization",
            "abs_magnetization", "site_magnetization",
            "max_force", "pressure", "nelec", "nbnd", "kmesh", "nkpt",
            "walltime", "job_id", "submitted_only",
        )
        out = {k: _jsonable(getattr(self, k)) for k in keys}
        out["workdir"] = str(self.workdir)
        out["files"] = dict(self.files)
        if self.messages:
            out["messages"] = self.messages
        return out

    def save_json(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.workdir / f"{self.task}_result.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        if self.structure is not None:
            payload["structure"] = self.structure.as_dict()
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target


class Engine:
    """計算バックエンドの基底クラス。

    バックエンド側で用意すべきなのは :meth:`run` のみ。設定の解釈、ジョブ投入、
    作図、ワークフロー制御は共通処理として提供される。
    """

    name = "base"
    supported: tuple[str, ...] = ()

    def __init__(self, config, scheduler=None) -> None:
        self.config = config
        self.scheduler = scheduler

    # -- 対応状況 ----------------------------------------------------------
    def supports(self, task: str) -> bool:
        return task in self.supported

    def check(self) -> list[str]:
        """このエンジンが実行できない理由を列挙して返す。"""
        return []

    # -- 実行 --------------------------------------------------------------
    def run(self, structure, task: str, workdir: Path, prev: CalcResult | None = None,
            **kwargs) -> CalcResult:
        raise NotImplementedError

    # -- 補助関数 ----------------------------------------------------------
    @staticmethod
    def _kmesh(config, structure) -> list[int]:
        from ezcal.structures import auto_kmesh

        mesh = config.get("dft.kmesh")
        if mesh:
            return [int(x) for x in mesh]
        return auto_kmesh(structure, float(config.get("dft.kspacing", 0.25)))
