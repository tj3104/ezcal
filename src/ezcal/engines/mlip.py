"""Machine-learning interatomic potential engine (SevenNet by default).

Only the tasks that a universal potential can actually answer are
supported: total energy, forces, stress and geometry optimisation.
Electronic-structure tasks (nscf / bands / dos) raise a clear error
instead of producing something meaningless.

Adding another MLIP means adding a branch to :meth:`MLIPEngine.calculator`;
nothing else in ezcal needs to change.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ezcal.engines.base import CalcResult, Engine, EngineError

EV_PER_ANG3_TO_GPA = 160.21766208


class MLIPEngine(Engine):
    """Single point energies and relaxations from a pre-trained potential."""

    name = "mlip"
    supported = ("scf", "relax", "vc-relax")

    def __init__(self, config, scheduler=None) -> None:
        super().__init__(config, scheduler)
        self._calc = None

    # ------------------------------------------------------------ set-up
    def check(self) -> list[str]:
        backend = str(self.config.get("mlip.backend", "sevennet")).lower()
        if backend in {"sevennet", "7net"}:
            try:
                import sevenn  # noqa: F401
            except ImportError:
                return ["SevenNet is not installed:  uv pip install sevenn"]
            return []
        return [f"unknown mlip backend {backend!r}"]

    def calculator(self):
        if self._calc is not None:
            return self._calc
        backend = str(self.config.get("mlip.backend", "sevennet")).lower()
        model = self.config.get("mlip.model", "7net-0")
        device = self.config.get("mlip.device", "cpu")
        if backend in {"sevennet", "7net"}:
            from sevenn.calculator import SevenNetCalculator

            self._calc = SevenNetCalculator(model=model, device=device)
        elif backend == "mace":                       # pragma: no cover - optional
            from mace.calculators import mace_mp

            self._calc = mace_mp(model=model, device=device)
        elif backend == "chgnet":                     # pragma: no cover - optional
            from chgnet.model.dynamics import CHGNetCalculator

            self._calc = CHGNetCalculator()
        else:
            raise EngineError(f"unknown mlip backend {backend!r}")
        return self._calc

    # ------------------------------------------------------------- runner
    def run(self, structure, task: str, workdir: Path, prev: CalcResult | None = None,
            **kwargs) -> CalcResult:
        task = task.lower()
        if not self.supports(task):
            raise EngineError(
                f"the MLIP engine cannot do {task!r}: a machine-learning potential has no "
                "electronic structure.  Use --engine qe for scf/nscf/bands/dos."
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
            f"{self.config.get('mlip.backend', 'sevennet')} model "
            f"{self.config.get('mlip.model', '7net-0')} on {self.config.get('mlip.device', 'cpu')}"
        )
        return result

    # -- helpers -----------------------------------------------------------
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
            except ImportError:                       # ASE < 3.23
                from ase.constraints import ExpCellFilter as Filter
        return Filter(atoms)
