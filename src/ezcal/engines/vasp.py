"""VASP engine.

The engine drives whatever ``vasp.command`` points at.  That is normally a
licensed ``vasp_std`` binary, but it can equally be ``mock-vasp`` from
aiida-vasp: a stand-in that hashes the parsed INCAR/KPOINTS/POSCAR of the
working directory, looks the hash up in a registry of finished
calculations, and copies the recorded outputs back.  That makes the whole
VASP path - input generation, execution, parsing, plotting - testable
without a VASP licence, and lets a group replay reference calculations.

Provenance is never hidden: a result that came out of the mock carries
``data["mock"]`` and says so in the run report.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ezcal.engines.base import CalcResult, Engine, EngineError
from ezcal.scheduler import Command, Stage

RY_EV = 13.605693122994
BOHR_ANG = 0.529177210903
KBAR_GPA = 0.1

#: how ezcal's engine-independent options map onto INCAR tags (all tasks)
INCAR_MAP: dict[str, tuple[str, Any]] = {
    "dft.ecutwfc": ("ENCUT", lambda ry: round(float(ry) * RY_EV, 1)),        # Ry -> eV
    "dft.conv_thr": ("EDIFF", lambda v: float(v) * RY_EV),                   # Ry -> eV
    "dft.degauss": ("SIGMA", lambda ry: round(float(ry) * RY_EV, 4)),        # Ry -> eV
    "dft.nbnd": ("NBANDS", int),
    "dft.electron_maxstep": ("NELM", int),
}

#: mapped only for the tasks that move atoms
RELAX_INCAR_MAP: dict[str, tuple[str, Any]] = {
    "relax.nstep": ("NSW", int),
    # Ry/Bohr -> eV/Angstrom, negative because VASP reads it as a force criterion
    "relax.forc_conv_thr": ("EDIFFG", lambda v: -abs(float(v) * RY_EV / BOHR_ANG)),
}

#: VASP's own sensible starting point, used when the ezcal option was left at
#: its packaged (Quantum ESPRESSO flavoured) default
VASP_DEFAULTS: dict[str, Any] = {"EDIFF": 1.0e-6}

#: ezcal task -> the INCAR settings that define it
TASK_INCAR: dict[str, dict[str, Any]] = {
    "scf": {"IBRION": -1, "NSW": 0, "ISIF": 2, "LCHARG": True, "LWAVE": True},
    "relax": {"IBRION": 2, "NSW": 100, "ISIF": 2, "LCHARG": True},
    "vc-relax": {"IBRION": 2, "NSW": 100, "ISIF": 3, "LCHARG": True},
    "nscf": {"IBRION": -1, "NSW": 0, "ICHARG": 11, "ISMEAR": -5, "LORBIT": 11},
    "bands": {"IBRION": -1, "NSW": 0, "ICHARG": 11, "ISMEAR": 0, "LORBIT": 11},
    "dos": {"IBRION": -1, "NSW": 0, "ICHARG": 11, "ISMEAR": -5, "LORBIT": 11},
}

#: occupation scheme -> ISMEAR
ISMEAR = {"smearing": 1, "gaussian": 0, "mv": 1, "mp": 1, "fd": -1,
          "fixed": 0, "tetrahedra": -5}

_MOCK_REGISTRY_RE = re.compile(r"Using test data in path (\S+) based detection")
_MOCK_CASE_RE = re.compile(r"Using test data from folder: (\S+)")
_MOCK_DEFAULT = "Using default test data"

PLACEHOLDER_POTCAR = """\
This is NOT a VASP pseudopotential.

ezcal wrote this placeholder because the configured VASP command is a mock
({command}) which only checks that the file exists.  A real VASP run needs a
licensed POTCAR library; point qe_config.yaml at it with

    vasp:
      potcar_dir: /path/to/potpaw_PBE
      potcar_mode: library

Elements in this calculation, in POSCAR order: {elements}
"""


class VaspEngine(Engine):
    """Run VASP (or aiida-vasp's ``mock-vasp``) and parse ``vasprun.xml``."""

    name = "vasp"
    supported = ("scf", "relax", "vc-relax", "nscf", "bands", "dos", "pdos")

    # ------------------------------------------------------------ command
    def command(self) -> str:
        return str(self.config.get("vasp.command", "vasp_std"))

    def is_mock(self) -> bool:
        """True when the configured command is aiida-vasp's mock executable."""
        return Path(self.command()).name.startswith("mock-vasp")

    def check(self) -> list[str]:
        import shutil

        command = self.command()
        problems: list[str] = []
        if shutil.which(command) is None and not Path(command).is_file():
            problems.append(
                f"VASP command {command!r} not found - set vasp.command in qe_config.yaml "
                "(use 'mock-vasp' from aiida-vasp to exercise the VASP path without a licence)"
            )
        if self.is_mock() and not self.config.get("vasp.mock.registry"):
            problems.append(
                "vasp.command is a mock but vasp.mock.registry is not set, so there is "
                "nothing to replay - point it at a registry directory"
            )
        return problems

    # ---------------------------------------------------------- INCAR
    def incar(self, structure, task: str) -> dict[str, Any]:
        """Assemble the INCAR tags for one task.

        ``vasp.incar`` overrides individual tags (a ``null`` value removes
        one); with ``vasp.incar_mode: replace`` it becomes the whole INCAR,
        which is what you want when reproducing a recorded calculation
        byte-for-byte.
        """
        overrides = dict(self.config.get("vasp.incar", {}) or {})
        if str(self.config.get("vasp.incar_mode", "merge")).lower() == "replace":
            return {key: value for key, value in overrides.items() if value is not None}

        cfg = self.config
        tags: dict[str, Any] = {"PREC": "Accurate", "LREAL": "Auto", "ALGO": "Normal"}
        tags.update(VASP_DEFAULTS)

        maps = dict(INCAR_MAP)
        if task in {"relax", "vc-relax"}:
            maps.update(RELAX_INCAR_MAP)
        for dotted, (tag, convert) in maps.items():
            value = cfg.get(dotted)
            # an untouched QE-flavoured default must not leak into VASP: conv_thr
            # 1e-8 Ry would become an EDIFF of 1.4e-7 eV, which is absurd here
            if value is None or (tag in VASP_DEFAULTS and not _is_user_set(dotted, value)):
                continue
            tags[tag] = convert(value)

        # the task defines the run type, so it wins over the generic mapping
        tags.update(TASK_INCAR.get(task, {}))

        if task in {"scf", "relax", "vc-relax"}:
            occupations = str(cfg.get("dft.occupations", "smearing"))
            smearing = str(cfg.get("dft.smearing", "mv"))
            tags["ISMEAR"] = ISMEAR.get(occupations if occupations != "smearing" else smearing, 1)

        if int(cfg.get("dft.nspin", 1) or 1) == 2:
            from ezcal.structures import label_element, site_labels

            tags["ISPIN"] = 2
            magmoms = cfg.get("dft.starting_magnetization", {}) or {}
            # VASP wants one starting moment per site, in POSCAR order and in Bohr
            # magneton.  QE reads the same option as a fraction of the valence
            # charge, so ezcal passes the number through unchanged rather than
            # inventing a conversion: what matters across both codes is the sign
            # pattern that sets up the magnetic order.  Sites left unspecified get
            # VASP's own default of 1.0.
            default = float(cfg.get("vasp.magmom_default", 1.0) or 1.0)
            per_site = []
            for label in site_labels(structure):
                value = magmoms.get(label, magmoms.get(label_element(label)))
                per_site.append(round(float(default if value is None else value), 4))
            tags["MAGMOM"] = " ".join(f"{v:g}" for v in per_site)

        hubbard = cfg.get("dft.hubbard_u", {}) or {}
        if hubbard:
            from ezcal.structures import label_element

            symbols = _poscar_species(structure)
            tags["LDAU"] = True
            tags["LDAUTYPE"] = 2
            tags["LDAUL"] = " ".join(
                str(_hubbard_l(label_element(s))) if _hubbard_value(hubbard, s) else "-1"
                for s in symbols)
            tags["LDAUU"] = " ".join(f"{_hubbard_value(hubbard, s) or 0.0:g}" for s in symbols)
            tags["LDAUJ"] = " ".join("0" for _ in symbols)
            tags["LMAXMIX"] = 4

        if cfg.get("dft.vdw_corr"):
            mapping = {"grimme-d3": {"IVDW": 12}, "grimme-d2": {"IVDW": 10},
                       "ts": {"IVDW": 20}}
            tags.update(mapping.get(str(cfg.get("dft.vdw_corr")).lower(), {}))

        for key, value in overrides.items():
            if value is None:
                tags.pop(key, None)
            else:
                tags[key] = value
        return tags

    # ---------------------------------------------------------- inputs
    def write_inputs(self, structure, task: str, workdir: Path,
                     kmesh: Sequence[int] | None = None, kpath=None) -> Path:
        from pymatgen.io.vasp.inputs import Poscar

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        Poscar(structure).write_file(str(workdir / "POSCAR"))
        # INCAR and KPOINTS are written here rather than through pymatgen so that
        # integers stay integers and the shift line is always present: mock-vasp
        # hashes the *parsed* input, and 200 and 200.0 are not the same value
        (workdir / "INCAR").write_text(render_incar(self.incar(structure, task)),
                                       encoding="utf-8")
        if kpath is not None:
            text = render_kpoints_explicit(np.asarray(kpath.kpoints, dtype=float))
        else:
            mesh = [int(v) for v in (kmesh or self._kmesh(self.config, structure))]
            shift = [float(v) for v in (self.config.get("dft.koffset") or [0, 0, 0])]
            style = str(self.config.get("vasp.kpoints_style", "gamma")).lower()
            text = render_kpoints_mesh(mesh, shift,
                                       "Monkhorst" if style.startswith("m") else "Gamma")
        (workdir / "KPOINTS").write_text(text, encoding="utf-8")

        self.write_potcar(structure, workdir)
        return workdir

    def write_potcar(self, structure, workdir: Path) -> Path:
        """Write POTCAR from a licensed library, or a labelled placeholder."""
        symbols = _poscar_species(structure)
        mode = str(self.config.get("vasp.potcar_mode", "auto")).lower()
        potcar_dir = self.config.path("vasp.potcar_dir")
        target = Path(workdir) / "POTCAR"

        if mode == "auto":
            mode = "library" if potcar_dir else ("placeholder" if self.is_mock() else "missing")

        if mode == "library":
            if not potcar_dir:
                raise EngineError("vasp.potcar_mode is 'library' but vasp.potcar_dir is not set")
            from pymatgen.io.vasp.inputs import Potcar

            mapping = self.config.get("vasp.potcar_map", {}) or {}
            names = [str(mapping.get(s, s)) for s in symbols]
            os.environ.setdefault("PMG_VASP_PSP_DIR", str(potcar_dir))
            potcar = Potcar(symbols=names,
                            functional=str(self.config.get("vasp.potcar_functional", "PBE")))
            potcar.write_file(str(target))
            return target

        if mode == "placeholder":
            target.write_text(
                PLACEHOLDER_POTCAR.format(command=self.command(), elements=", ".join(symbols)),
                encoding="utf-8")
            return target

        raise EngineError(
            "no POTCAR available: set vasp.potcar_dir to a licensed library, or use a mock "
            "command (vasp.command: mock-vasp) which does not need one"
        )

    # ---------------------------------------------------------- execution
    def registry_path(self) -> Path | None:
        """The mock registry directory, or None.

        mock-vasp 5.1 takes a single base path (it calls ``Path()`` on whatever
        ``MOCK_VASP_REG_BASE`` holds), so a list in the config uses its first entry.
        """
        registry = self.config.get("vasp.mock.registry")
        if not registry:
            return None
        if isinstance(registry, (list, tuple)):
            registry = registry[0]
        return Path(os.path.expandvars(str(registry))).expanduser().resolve()

    def stage_env(self) -> dict[str, str]:
        """Environment for the mock code, empty for a real VASP run."""
        env: dict[str, str] = {}
        registry = self.registry_path()
        if registry:
            env["MOCK_VASP_REG_BASE"] = str(registry)
        vasp_cmd = self.config.get("vasp.mock.vasp_cmd")
        if vasp_cmd:
            env["MOCK_VASP_VASP_CMD"] = str(vasp_cmd)
        return env

    def run(self, structure, task: str, workdir: Path, prev: CalcResult | None = None,
            **kwargs) -> CalcResult:
        task = task.lower()
        if not self.supports(task):
            raise EngineError(f"the VASP engine cannot run task {task!r}")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        kpath = kwargs.get("kpath")
        if task == "bands" and kpath is None:
            from ezcal.structures import band_path

            kpath = band_path(
                structure,
                line_density=float(self.config.get("bands.line_density", 25)),
                symprec=float(self.config.get("bands.symprec", 1e-5)),
                min_points=int(self.config.get("bands.min_points_per_segment", 6)),
            )
        self.write_inputs(structure, task, workdir, kmesh=kwargs.get("kmesh"), kpath=kpath)

        # a band/dos run reads the charge density of the previous scf
        if task in {"nscf", "bands", "dos"} and prev is not None:
            _link_chgcar(Path(prev.workdir), workdir)

        stage = Stage(
            name=task,
            workdir=workdir,
            env=self.stage_env(),
            commands=[Command(self.command(), [], stdout=workdir / "vasp.out",
                              label=f"vasp {task}")],
        )
        job = self.scheduler.execute(stage)

        result = CalcResult(task=task, engine=self.name, workdir=workdir,
                            job_id=job.job_id, submitted_only=job.submitted_only,
                            walltime=job.elapsed)
        for name in ("INCAR", "POSCAR", "KPOINTS"):
            result.files[name.lower()] = str(workdir / name)
        result.files["output"] = str(workdir / "vasp.out")
        if job.script:
            result.files["job_script"] = str(job.script)

        if job.submitted_only:
            result.ok = True
            result.messages.append(
                f"submitted as {job.job_id}" if job.job_id
                else "dry run: VASP inputs written, nothing executed")
            return result
        if not job.ok:
            result.messages += job.log[-2:]
            result.messages.append(f"VASP failed (exit code {job.returncode}); see {workdir}")
            return result

        self._note_mock(result, workdir)
        self._parse(result, workdir, structure, kpath)
        return result

    # ---------------------------------------------------------- provenance
    def _note_mock(self, result: CalcResult, workdir: Path) -> None:
        if not self.is_mock():
            return
        log = workdir / "vasp_output"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        info: dict[str, Any] = {"used": True, "command": self.command()}

        match = _MOCK_REGISTRY_RE.search(text)
        case = _MOCK_CASE_RE.search(text)
        if match:
            info.update(mode="registry", source=match.group(1))
            result.messages.append(
                f"mock-vasp replayed recorded VASP output from {match.group(1)} "
                "(the inputs hash-matched that entry)")
        elif case:
            info.update(mode="test-case", source=case.group(1))
            result.messages.append(f"mock-vasp used the bundled test case {case.group(1)!r}")
        elif _MOCK_DEFAULT in text:
            info.update(mode="default", source=None)
            result.messages.append(
                "WARNING: mock-vasp fell back to its built-in demo data, which describes a "
                "DIFFERENT system - these numbers are not a calculation of your structure")
        else:
            info.update(mode="unknown", source=None)
            result.messages.append("mock-vasp was used; provenance could not be determined")
        result.data["mock"] = info

    # ---------------------------------------------------------- parsing
    def _parse(self, result: CalcResult, workdir: Path, structure, kpath) -> None:
        from pymatgen.io.vasp.outputs import Vasprun

        xml = workdir / "vasprun.xml"
        if not xml.is_file():
            result.messages.append(f"vasprun.xml not found in {workdir}")
            return
        try:
            # the DOS block also carries efermi, so it is always worth reading
            run = Vasprun(str(xml), parse_potcar_file=False, parse_dos=True, parse_eigen=True)
        except Exception as exc:                       # malformed / truncated run
            result.messages.append(f"could not parse vasprun.xml: {exc}")
            return

        result.files["vasprun"] = str(xml)
        result.converged = bool(run.converged)
        result.ok = True

        if result.task not in {"nscf", "bands"}:
            result.energy = float(run.final_energy)
            result.energy_per_atom = result.energy / max(1, len(run.final_structure))
        result.fermi_energy = float(run.efermi) if run.efermi is not None else None
        result.structure = run.final_structure
        result.nbnd = int(run.parameters.get("NBANDS", 0)) or None
        result.nelec = float(run.parameters.get("NELECT", 0)) or None

        steps = run.ionic_steps or []
        if steps:
            last = steps[-1]
            if last.get("forces") is not None:
                forces = np.asarray(last["forces"], dtype=float)
                result.forces = forces.tolist()
                result.max_force = float(np.abs(forces).max())
            if last.get("stress") is not None:
                stress = np.asarray(last["stress"], dtype=float) * KBAR_GPA   # kBar -> GPa
                result.stress = stress.tolist()
                result.pressure = float(np.trace(stress) / 3.0)
            energies = [s["e_wo_entrp"] for s in steps if s.get("e_wo_entrp") is not None]
            if len(energies) > 1:
                result.data["scf_energies"] = [float(e) for e in energies]

        self._parse_magnetism(result, workdir, run)
        self._parse_eigenvalues(result, run, kpath)
        # unlike QE, VASP writes the DOS into vasprun.xml for every run that has one
        self._parse_dos(result, run)

    def _parse_magnetism(self, result: CalcResult, workdir: Path, run) -> None:
        if int(run.parameters.get("ISPIN", 1)) != 2:
            return
        outcar = workdir / "OUTCAR"
        if not outcar.is_file():
            return
        from pymatgen.io.vasp.outputs import Outcar

        try:
            parsed = Outcar(str(outcar))
        except Exception:
            return
        if parsed.total_magnetization is not None:
            result.magnetization = float(parsed.total_magnetization)
        moments = [float(entry.get("tot", 0.0)) for entry in (parsed.magnetization or [])]
        if moments:
            result.site_magnetization = moments
            result.abs_magnetization = float(np.abs(moments).sum())

    def _parse_eigenvalues(self, result: CalcResult, run, kpath) -> None:
        from pymatgen.electronic_structure.core import Spin

        if not run.eigenvalues:
            return
        spins = [Spin.up] + ([Spin.down] if Spin.down in run.eigenvalues else [])
        eig = np.stack([np.asarray(run.eigenvalues[s])[:, :, 0] for s in spins])  # (ns, nk, nb)
        occ = np.stack([np.asarray(run.eigenvalues[s])[:, :, 1] for s in spins])
        result.data["eigenvalues"] = eig
        result.data["occupations"] = occ
        result.data["kpoints_frac"] = np.asarray(run.actual_kpoints, dtype=float)
        result.nkpt = int(eig.shape[1])

        gap = _band_gap(eig, occ, result.fermi_energy)
        if gap is not None:
            result.band_gap = gap["gap"]
            result.data["gap_info"] = gap
            if not gap["metal"]:
                result.homo, result.lumo = gap["vbm"], gap["cbm"]

        if result.task == "bands" and kpath is not None:
            result.data["kpath"] = {
                "distances": kpath.distances.tolist(),
                "labels": [[int(i), label] for i, label in kpath.labels],
                "kpoints": kpath.kpoints.tolist(),
            }

    def _parse_dos(self, result: CalcResult, run) -> None:
        from pymatgen.electronic_structure.core import Orbital, Spin

        try:
            dos = run.complete_dos
        except Exception:
            dos = getattr(run, "tdos", None)
        if dos is None:
            return

        energies = np.asarray(dos.energies, dtype=float)
        densities = dos.densities
        payload: dict[str, Any] = {"energy": energies,
                                   "fermi_energy": float(dos.efermi) if dos.efermi else None}
        if Spin.down in densities:
            payload["dos_up"] = np.asarray(densities[Spin.up], dtype=float)
            payload["dos_down"] = np.asarray(densities[Spin.down], dtype=float)
            payload["dos"] = payload["dos_up"] + payload["dos_down"]
        else:
            payload["dos"] = np.asarray(densities[Spin.up], dtype=float)
        result.data["dos"] = payload

        if not hasattr(dos, "get_element_spd_dos"):
            return
        per_element: dict[str, np.ndarray] = {}
        per_orbital: dict[str, np.ndarray] = {}
        try:
            for element, element_dos in dos.get_element_dos().items():
                per_element[str(element)] = _sum_spins(element_dos.densities)
            for element in dos.structure.composition.elements:
                for orbital, orbital_dos in dos.get_element_spd_dos(element).items():
                    key = f"{element}-{str(orbital).lower()}"
                    per_orbital[key] = _sum_spins(orbital_dos.densities)
        except Exception:
            pass
        if per_element or per_orbital:
            result.data["pdos"] = {"energy": energies, "spin_polarised": Spin.down in densities,
                                   "per_element": per_element, "per_orbital": per_orbital,
                                   "total": payload["dos"]}

    # ------------------------------------------------------------ registry
    def record(self, rundir: Path, name: str) -> Path:
        """Add a finished VASP run to the mock registry so it can be replayed."""
        from aiida_vasp.utils.mock_code import VaspMockRegistry

        base = self.registry_path()
        if base is None:
            raise EngineError("vasp.mock.registry is not set")
        base.mkdir(parents=True, exist_ok=True)

        rundir = Path(rundir)
        missing = [f for f in ("INCAR", "POSCAR", "KPOINTS", "vasprun.xml")
                   if not (rundir / f).is_file()]
        if missing:
            raise EngineError(f"{rundir} is not a finished VASP run (missing {', '.join(missing)})")

        reg = VaspMockRegistry(str(base))
        reg.upload_calc(rundir, name)
        return base / name


def _is_user_set(dotted: str, value: Any) -> bool:
    """True when ``value`` differs from the value shipped in default_config.yaml."""
    from ezcal.config import default_config

    try:
        shipped = default_config().get(dotted)
    except Exception:
        return True
    return shipped != value


# ------------------------------------------------------------- input writers
def incar_value(value: Any) -> str:
    """Render one INCAR value, keeping ints as ints and bools Fortran-style."""
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(incar_value(v) for v in value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.10g}"
        return text if any(c in text for c in ".eE") else text + ".0"
    return str(value)


def render_incar(tags: dict[str, Any]) -> str:
    return "".join(f"{key} = {incar_value(value)}\n"
                   for key, value in sorted(tags.items()) if value is not None)


def render_kpoints_mesh(mesh: Sequence[int], shift: Sequence[float] = (0, 0, 0),
                        centering: str = "Gamma") -> str:
    return (
        "Automatic mesh written by ezcal\n"
        "0\n"
        f"{centering}\n"
        "  " + "  ".join(f"{int(v):d}" for v in mesh) + "\n"
        "  " + "  ".join(f"{float(v):.9f}" for v in shift) + "\n"
    )


def render_kpoints_explicit(points: np.ndarray, weight: float = 1.0) -> str:
    lines = ["ezcal band path", f"{len(points)}", "Reciprocal"]
    for kx, ky, kz in points:
        lines.append(f"  {kx:14.10f} {ky:14.10f} {kz:14.10f} {weight:8.4f}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ helpers
def _poscar_species(structure) -> list[str]:
    """Element symbols in POSCAR order, deduplicated the way VASP groups them."""
    from pymatgen.io.vasp.inputs import Poscar

    return [str(s) for s in Poscar(structure).site_symbols]


def _hubbard_value(hubbard, symbol):
    from ezcal.structures import label_element

    value = hubbard.get(symbol, hubbard.get(label_element(symbol)))
    return float(value) if value is not None else None


def _hubbard_l(element: str) -> int:
    from pymatgen.core.periodic_table import Element

    return {"s": 0, "p": 1, "d": 2, "f": 3}.get(Element(element).block, 2)


def _sum_spins(densities) -> np.ndarray:
    return np.asarray(sum(np.asarray(v, dtype=float) for v in densities.values()))


def _link_chgcar(source: Path, target: Path) -> None:
    """Carry CHGCAR/WAVECAR from the previous step into this one."""
    import shutil

    for name in ("CHGCAR", "WAVECAR"):
        origin = Path(source) / name
        if origin.is_file() and not (Path(target) / name).exists():
            try:
                os.link(origin, Path(target) / name)
            except OSError:
                shutil.copy2(origin, Path(target) / name)


def _band_gap(eig: np.ndarray, occ: np.ndarray, fermi: float | None,
              tol: float = 1e-3) -> dict | None:
    """Same electron-counting rule as the QE engine, on VASP occupations."""
    nspin, nkpt, nbnd = eig.shape
    filled = occ > 0.5
    if not filled.any() or filled.all():
        return None
    counts = {int(filled[s, k].sum()) for s in range(nspin) for k in range(nkpt)}
    if len(counts) != 1:                                # a band crosses E_F
        return {"gap": 0.0, "metal": True, "vbm": None, "cbm": None, "direct": None}
    nocc = counts.pop()
    if nocc < 1 or nocc >= nbnd:
        return None
    homo_k = eig[:, :, nocc - 1].max(axis=0)
    lumo_k = eig[:, :, nocc].min(axis=0)
    vbm, cbm = float(homo_k.max()), float(lumo_k.min())
    if cbm - vbm <= tol:
        return {"gap": 0.0, "metal": True, "vbm": vbm, "cbm": cbm, "direct": None}
    return {"gap": cbm - vbm, "metal": False, "vbm": vbm, "cbm": cbm,
            "direct": bool(int(np.argmax(homo_k)) == int(np.argmin(lumo_k))),
            "k_vbm": int(np.argmax(homo_k)), "k_cbm": int(np.argmin(lumo_k))}
