"""機械学習原子間ポテンシャル (MLIP / NNP) エンジン (既定は SevenNet)。

汎用ポテンシャルが実際に答えられるタスクだけを対象とする。すなわち全エネルギー、
力、応力、構造最適化である。電子構造に関わるタスク (nscf / bands / dos) は、
無意味な結果を返す代わりに明示的なエラーを送出する。

calculator そのものの作り方はこのモジュールには書かれておらず、
:mod:`ezcal.calculators` が解決する。ASE の calculator を返せるなら、
同梱レシピ (``--mlip-backend sevennet`` 等)、import パス (``mlip.factory``)、
ユーザーの Python スクリプト (``mlip.script``) のどれでも同じように動く。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ezcal.engines.base import CalcResult, Engine, EngineError

EV_PER_ANG3_TO_GPA = 160.21766208


class MLIPEngine(Engine):
    """学習済みポテンシャルによる一点計算と構造最適化。"""

    name = "mlip"
    supported = ("scf", "relax", "vc-relax")

    def __init__(self, config, scheduler=None) -> None:
        super().__init__(config, scheduler)
        self._calc = None

    # ------------------------------------------------------------ 準備処理
    def check(self) -> list[str]:
        from ezcal import calculators

        return calculators.check(self.config)

    def describe(self) -> str:
        from ezcal import calculators

        return calculators.describe(self.config)

    def calculator(self):
        """設定が指すポテンシャルの ASE calculator (1 度作ったら使い回す)。"""
        if self._calc is None:
            from ezcal import calculators
            from ezcal.calculators import CalculatorError

            try:
                self._calc = calculators.get_calculator(self.config)
            except CalculatorError as exc:
                raise EngineError(str(exc)) from exc
        return self._calc

    # ------------------------------------------------------------- 実行部
    def run(self, structure, task: str, workdir: Path, prev: CalcResult | None = None,
            **kwargs) -> CalcResult:
        task = task.lower()
        if not self.supports(task):
            raise EngineError(
                f"MLIP エンジンは {task!r} を実行できません。機械学習ポテンシャルは電子構造を"
                "持たないためです。scf/nscf/bands/dos には --engine qe を使ってください。"
            )
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        from ezcal.structures import from_ase, to_ase

        atoms = to_ase(structure)
        atoms.calc = self.calculator()
        start = time.time()
        result = CalcResult(task=task, engine=self.name, workdir=workdir)

        if task in {"relax", "vc-relax"}:
            traj_path = workdir / f"{task}.traj"
            log_path = workdir / f"{task}.log"
            target = atoms
            if task == "vc-relax":
                target = self._cell_filter(atoms)
            optimizer = self._optimizer(target, str(log_path), str(traj_path))
            optimizer.run(fmax=float(self.config.get("mlip.fmax", 0.02)),
                          steps=int(self.config.get("mlip.steps", 300)))
            result.converged = bool(optimizer.converged())
            result.files["log"] = str(log_path)
            result.files["trajectory"] = str(traj_path)
        else:
            result.converged = True

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())
        result.energy = energy
        result.energy_per_atom = energy / max(1, len(atoms))
        result.forces = forces.tolist()
        result.max_force = float(np.abs(forces).max())
        try:
            stress = atoms.get_stress(voigt=False) * EV_PER_ANG3_TO_GPA
            result.stress = np.asarray(stress).tolist()
            result.pressure = float(np.trace(np.asarray(stress)) / 3.0)
        except Exception:
            pass
        try:
            magmoms = atoms.get_magnetic_moments()
            result.magnetization = float(np.sum(magmoms))
        except Exception:
            pass

        result.structure = from_ase(atoms)
        result.walltime = time.time() - start
        result.ok = True
        result.messages.append(
            f"MLIP: {self.describe()} on {self.config.get('mlip.device', 'cpu')}")
        return result

    # -- 補助関数 ----------------------------------------------------------
    def _optimizer(self, target, logfile: str, trajectory: str):
        name = str(self.config.get("mlip.optimizer", "FIRE")).upper()
        if name == "BFGS":
            from ase.optimize import BFGS as Opt
        elif name == "LBFGS":
            from ase.optimize import LBFGS as Opt
        else:
            from ase.optimize import FIRE as Opt
        return Opt(target, logfile=logfile, trajectory=trajectory)

    def _cell_filter(self, atoms):
        kind = str(self.config.get("mlip.filter", "frechet")).lower()
        if kind == "exp":
            from ase.filters import ExpCellFilter as Filter
        elif kind == "unit":
            from ase.filters import UnitCellFilter as Filter
        else:
            try:
                from ase.filters import FrechetCellFilter as Filter
            except ImportError:                       # ASE 3.23 未満
                from ase.constraints import ExpCellFilter as Filter
        return Filter(atoms)
