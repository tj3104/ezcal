"""Task chains.

``ezcal bands Si.cif`` has to work on a bare structure, so every task
carries the list of steps it depends on and the workflow runs them in
order, reusing the charge density through a single shared QE ``outdir``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ezcal.engines import get_engine
from ezcal.engines.base import CalcResult
from ezcal.scheduler import get_scheduler

#: task -> the steps that have to run before it (in order)
CHAINS: dict[str, tuple[str, ...]] = {
    "scf": ("scf",),
    "relax": ("relax",),
    "vc-relax": ("vc-relax",),
    "nscf": ("scf", "nscf"),
    "dos": ("scf", "nscf", "dos"),
    "pdos": ("scf", "nscf", "dos"),
    "bands": ("scf", "bands"),
    "charge": ("scf", "dos", "charge"),      # dos first: projwfc gives the Loewdin charges
    "auto": ("vc-relax", "scf", "nscf", "dos", "bands"),
}

STEP_TITLES = {
    "relax": "ionic relaxation",
    "vc-relax": "cell + ionic relaxation",
    "scf": "self-consistent field",
    "nscf": "non self-consistent (dense mesh)",
    "dos": "density of states",
    "bands": "band structure",
    "charge": "charge density and atomic charges",
}


@dataclass
class WorkflowResult:
    task: str
    rundir: Path
    steps: dict[str, CalcResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)
    exports: list[Path] = field(default_factory=list)
    structure_initial: Any = None
    structure_final: Any = None
    ok: bool = False
    submitted_only: bool = False
    elapsed: float = 0.0
    messages: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    @property
    def last(self) -> CalcResult | None:
        return self.steps[self.order[-1]] if self.order else None

    def summary(self) -> dict:
        from ezcal.structures import structure_info

        payload: dict[str, Any] = {
            "task": self.task,
            "ok": self.ok,
            "rundir": str(self.rundir),
            "elapsed_s": round(self.elapsed, 2),
            "steps": {name: self.steps[name].summary() for name in self.order},
            "plots": [str(p) for p in self.plots],
            "exports": [str(p) for p in self.exports],
            "messages": self.messages,
        }
        if self.structure_initial is not None:
            payload["structure_initial"] = structure_info(self.structure_initial)
        if self.structure_final is not None:
            payload["structure_final"] = structure_info(self.structure_final)
            payload["structure_final_cif"] = str(self.rundir / "final_structure.cif")
        return payload


class Workflow:
    """Run a task (and everything it needs) for one structure."""

    def __init__(self, config, structure, rundir: str | Path, label: str = "",
                 log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.structure = self._apply_sublattices(config, structure,
                                                 log or (lambda msg: None))
        self.rundir = Path(rundir)
        self.label = label or structure.composition.reduced_formula
        self.log = log or (lambda msg: None)
        self.scheduler = get_scheduler(config)
        self.engine = get_engine(config.get("engine", "qe"), config, self.scheduler)
        self.outdir = (self.rundir / "tmp").resolve()

    @staticmethod
    def _apply_sublattices(config, structure, log):
        """Turn ``dft.magnetic_sublattices`` into pw.x species labels.

        ``{"Fe": [0.6, -0.6]}`` labels the Fe sites Fe1, Fe2, Fe1, ... in the
        order they appear, so an antiferromagnet is two species as far as
        pw.x is concerned and the symmetry is lowered accordingly.
        """
        from ezcal.structures import split_sublattices

        spec = config.get("dft.magnetic_sublattices") or {}
        spec = {element: values for element, values in spec.items() if values}
        if not spec:
            return structure

        labelled, magnetization = split_sublattices(structure, spec)
        config.set("dft.nspin", 2)
        merged = dict(config.get("dft.starting_magnetization") or {})
        merged.update(magnetization)
        config.set("dft.starting_magnetization", merged)
        summary = ", ".join(f"{label} {value:+g}"
                            for label, value in sorted(magnetization.items()))
        log(f"    magnetic sublattices: {summary}")
        return labelled

    # ---------------------------------------------------------------- plan
    def plan(self, task: str, skip: Sequence[str] = (), only: bool = False) -> list[str]:
        task = task.lower()
        chain = list(CHAINS.get(task, (task,)))
        if only:
            chain = [task] if task != "auto" else chain
        chain = [step for step in chain if step not in skip]
        return [step for step in chain if self.engine.supports(step)]

    # ----------------------------------------------------------------- run
    def run(self, task: str, skip: Sequence[str] = (), only: bool = False) -> WorkflowResult:
        start = time.time()
        result = WorkflowResult(task=task, rundir=self.rundir,
                                structure_initial=self.structure)
        if task not in CHAINS or task != "auto":
            wanted = task if task in CHAINS else task
            if wanted != "auto" and not self.engine.supports(wanted):
                result.messages.append(
                    f"engine {self.engine.name!r} cannot run {wanted!r} "
                    f"(it supports: {', '.join(self.engine.supported)})")
                result.elapsed = time.time() - start
                return result
        steps = self.plan(task, skip=skip, only=only)
        if task == "auto" and len(steps) < len(CHAINS["auto"]):
            dropped = [s for s in CHAINS["auto"] if s not in steps and s not in skip]
            if dropped:
                result.messages.append(
                    f"engine {self.engine.name!r} skipped unsupported steps: "
                    + ", ".join(dropped))
        if not steps:
            result.messages.append(
                f"engine {self.engine.name!r} supports none of the steps needed for {task!r} "
                f"(supported: {', '.join(self.engine.supported)})"
            )
            result.elapsed = time.time() - start
            return result

        self.rundir.mkdir(parents=True, exist_ok=True)
        self._save_structure(self.structure, "input_structure")

        problems = self.engine.check()
        if problems:
            result.messages += problems
            result.elapsed = time.time() - start
            return result

        structure = self.structure
        kpath = None
        prev: CalcResult | None = None

        for index, step in enumerate(steps):
            # the band path fixes the cell: switch to the seekpath primitive
            # cell before the scf so that every later step shares it
            if "bands" in steps and step == "scf" and kpath is None:
                kpath, structure = self._prepare_band_path(structure)

            workdir = self.rundir / f"{index:02d}_{step}"
            self.log(f"[{index + 1}/{len(steps)}] {step}: {STEP_TITLES.get(step, step)}")
            kwargs: dict[str, Any] = {"outdir": self.outdir}
            if step == "nscf":
                kwargs["kmesh"] = self._nscf_mesh(structure)
                kwargs["occupations"] = self.config.get("nscf.occupations", "tetrahedra")
            if step == "bands" and kpath is not None:
                kwargs["kpath"] = kpath

            step_result = self.engine.run(structure, step, workdir, prev=prev, **kwargs)
            result.steps[step] = step_result
            result.order.append(step)
            step_result.save_json()
            for note in step_result.messages:
                result.messages.append(f"[{step}] {note}")

            if step_result.submitted_only:
                result.submitted_only = True
                result.ok = True
                if step_result.job_id:
                    result.messages.append(
                        f"step {step!r} was submitted as job {step_result.job_id}; "
                        "add --qsub-wait to chain the remaining steps automatically")
                else:
                    remaining = ", ".join(steps[index + 1:]) or "none"
                    result.messages.append(
                        f"dry run: inputs and job script for {step!r} were written, "
                        f"nothing was executed (remaining steps: {remaining})")
                break
            if not step_result.ok:
                result.messages.append(f"step {step!r} failed")
                break

            if step_result.structure is not None and step in {"relax", "vc-relax"}:
                structure = step_result.structure
                self._save_structure(structure, "relaxed_structure")
            prev = step_result
        else:
            result.ok = True

        result.structure_final = structure
        if result.structure_final is not None:
            self._save_structure(result.structure_final, "final_structure")
        result.elapsed = time.time() - start

        if not result.submitted_only:
            self._postprocess(result)
        self._write_summary(result)
        self._cleanup()
        return result

    # ----------------------------------------------------------- internals
    def _prepare_band_path(self, structure):
        from ezcal.structures import band_path

        kpath = band_path(
            structure,
            line_density=float(self.config.get("bands.line_density", 25)),
            symprec=float(self.config.get("bands.symprec", 1e-5)),
            min_points=int(self.config.get("bands.min_points_per_segment", 6)),
        )
        primitive = kpath.primitive_structure
        if primitive is not None and len(primitive) != len(structure):
            self.log(f"    band path: using the seekpath primitive cell "
                     f"({len(structure)} -> {len(primitive)} atoms)")
        path_text = " -> ".join(dict.fromkeys(
            [lab for _, lab in kpath.labels if lab]))
        self.log(f"    k-path ({kpath.nkpt} points): {path_text}")
        return kpath, primitive if primitive is not None else structure

    def _nscf_mesh(self, structure) -> list[int]:
        from ezcal.structures import scale_kmesh

        base = self.engine._kmesh(self.config, structure)
        return scale_kmesh(base, float(self.config.get("nscf.kmesh_scale", 2)))

    def _save_structure(self, structure, name: str) -> None:
        try:
            structure.to(filename=str(self.rundir / f"{name}.cif"))
            (self.rundir / f"{name}.json").write_text(
                json.dumps(structure.as_dict(), indent=1, default=str), encoding="utf-8")
        except Exception as exc:                       # never fail a run over this
            self.log(f"    (could not write {name}: {exc})")

    def _postprocess(self, result: WorkflowResult) -> None:
        from ezcal import plotting

        backends = plotting.resolve_backends(self.config.get("output.plot"))
        if not backends:
            return
        plots_dir = self.rundir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(self.config.get("output.dpi", 200))

        fermi = None
        zero = "F"
        for name in ("bands", "nscf", "scf"):
            step = result.steps.get(name)
            if step is None:
                continue
            gap_info = step.data.get("gap_info") or {}
            if gap_info and not gap_info.get("metal") and gap_info.get("vbm") is not None:
                fermi = gap_info["vbm"]      # zero at the valence band maximum
                zero = "VBM"
                break
            if step.fermi_energy is not None:
                fermi = step.fermi_energy
                break

        bands = result.steps.get("bands")
        dos = result.steps.get("dos") or next(
            (result.steps[name] for name in reversed(result.order)
             if result.steps[name].ok and result.steps[name].data.get("dos")), None)
        self._save_raw(bands, dos, fermi, zero)
        emin = self.config.get("bands.emin")
        emax = self.config.get("bands.emax")
        emin = float(emin) if emin is not None else -10.0
        emax = float(emax) if emax is not None else 10.0

        if bands is not None and bands.ok and "eigenvalues" in bands.data:
            result.plots += plotting.plot_bands(
                bands, plots_dir, backends, fermi=fermi, emin=emin, emax=emax,
                dpi=dpi, title=f"{self.label} band structure", zero=zero)
            result.exports.append(
                plotting.export_bands_csv(bands, plots_dir / "bands.csv", fermi))
        if dos is not None and dos.ok:
            result.plots += plotting.plot_dos(
                dos, plots_dir, backends, fermi=fermi, emin=emin, emax=emax,
                dpi=dpi, title=f"{self.label} density of states", zero=zero)
            csv = plotting.export_dos_csv(dos, plots_dir / "dos.csv", fermi)
            if csv:
                result.exports.append(csv)
        if bands is not None and dos is not None and bands.ok and dos.ok:
            result.plots += plotting.plot_bands_dos(
                bands, dos, plots_dir, backends, fermi=fermi, emin=emin, emax=emax,
                dpi=dpi, title=self.label, zero=zero)
        result.plots += plotting.plot_convergence(result.steps, plots_dir, backends, dpi)
        self._plot_charge(result, plots_dir, backends, dpi)

    def _plot_charge(self, result: WorkflowResult, plots_dir: Path,
                     backends: Sequence[str], dpi: int) -> None:
        """Profiles, a slice and a 3D isosurface for every cube that was written."""
        from ezcal import plotting
        from ezcal.charge import read_cube

        step = result.steps.get("charge")
        if step is None or not step.ok:
            return
        raw_axis = self.config.get("charge.slice_axis")
        axis = None if raw_axis is None else int(raw_axis)
        fraction = float(self.config.get("charge.slice_fraction", 0.5))
        for key, path in step.files.items():
            if not key.startswith("cube_"):
                continue
            kind = key[len("cube_"):]
            try:
                cube = read_cube(path)
            except Exception as exc:
                result.messages.append(f"could not plot {kind}: {exc}")
                continue
            label = f"{self.label} {kind}"
            result.plots += plotting.plot_charge_profile(
                cube, plots_dir, backends, kind=kind, dpi=dpi, title=label)
            result.plots += plotting.plot_charge_slice(
                cube, plots_dir, backends, kind=kind, axis=axis, fraction=fraction,
                dpi=dpi, title=label)
            if self.config.get("charge.isosurface", True) and "plotly" in backends:
                result.plots += plotting.plot_charge_isosurface(
                    cube, plots_dir, kind=kind,
                    max_points=int(self.config.get("charge.isosurface_grid", 64)),
                    title=label)
            if not self.config.get("charge.keep_cube", True):
                Path(path).unlink(missing_ok=True)

    def _save_raw(self, bands, dos, fermi, zero: str = "F") -> None:
        """Dump the band/DOS arrays so `ezcal plot` can redraw without QE."""
        import numpy as np

        payload: dict[str, Any] = {"fermi": fermi, "zero": zero}
        if bands is not None and bands.ok and bands.data.get("eigenvalues") is not None:
            payload["bands"] = {
                "eigenvalues": np.asarray(bands.data["eigenvalues"]).tolist(),
                "kpath": bands.data.get("kpath", {}),
            }
        if dos is not None and dos.ok and dos.data.get("dos"):
            raw = dos.data["dos"]
            payload["dos"] = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") else v)
                              for k, v in raw.items() if v is not None}
            pdos = dos.data.get("pdos") or {}
            if pdos:
                payload["pdos"] = {
                    key: ({k: np.asarray(v).tolist() for k, v in value.items()}
                          if isinstance(value, dict)
                          else (np.asarray(value).tolist()
                                if hasattr(value, "__len__") else value))
                    for key, value in pdos.items() if value is not None
                }
        if len(payload) > 1:
            (self.rundir / "raw_data.json").write_text(
                json.dumps(payload), encoding="utf-8")

    def _write_properties(self, result: WorkflowResult) -> dict:
        """The physical quantities, gathered in one place.

        Band gap, Fermi level and the per-atom magnetic moments and charges are
        spread over several steps; this pulls them together so nobody has to
        know which step produced what.
        """
        from ezcal.structures import site_labels

        dense = next((result.steps[n] for n in ("bands", "nscf", "scf")
                      if n in result.steps and result.steps[n].ok), None)
        scf = result.steps.get("scf") or (result.steps.get(result.order[0])
                                          if result.order else None)
        charge = result.steps.get("charge")

        payload: dict[str, Any] = {"formula": None, "task": result.task,
                                   "rundir": str(self.rundir)}
        if result.structure_final is not None:
            payload["formula"] = result.structure_final.composition.reduced_formula

        if dense is not None:
            gap_info = dense.data.get("gap_info") or {}
            payload["electronic"] = {
                "fermi_energy_eV": dense.fermi_energy,
                "band_gap_eV": dense.band_gap,
                "metal": gap_info.get("metal"),
                "gap_kind": None if gap_info.get("metal") in (None, True) else
                            ("direct" if gap_info.get("direct") else "indirect"),
                "vbm_eV": gap_info.get("vbm"),
                "cbm_eV": gap_info.get("cbm"),
                "nelec": dense.nelec,
                "nbnd": dense.nbnd,
                "source_step": dense.task,
            }
        if scf is not None:
            payload["magnetism"] = {
                "nspin": int(self.config.get("dft.nspin", 1) or 1),
                "total_magnetization_uB": scf.magnetization,
                "absolute_magnetization_uB": scf.abs_magnetization,
            }
        if result.structure_final is not None:
            labels = site_labels(result.structure_final)
            moments = (scf.site_magnetization if scf is not None else None) or []
            charge_rows = {row["index"]: row for row in
                           ((charge.data.get("atoms") if charge else None) or [])}
            atoms = []
            for index, site in enumerate(result.structure_final):
                entry: dict[str, Any] = {
                    "index": index + 1, "label": labels[index],
                    "element": site.specie.symbol,
                    "moment_sphere_uB": moments[index] if index < len(moments) else None,
                }
                extra = charge_rows.get(index + 1, {})
                for key, out in (("lowdin_moment", "moment_lowdin_uB"),
                                 ("lowdin_charge", "lowdin_charge_e"),
                                 ("lowdin_electrons", "lowdin_electrons"),
                                 ("bader_charge", "bader_charge_e"),
                                 ("bader_electrons", "bader_electrons"),
                                 ("bader_volume", "bader_volume_A3")):
                    if extra.get(key) is not None:
                        entry[out] = extra[key]
                atoms.append(entry)
            payload["atoms"] = atoms

        (self.rundir / "properties.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (self.rundir / "properties.md").write_text(
            render_properties(payload), encoding="utf-8")
        return payload

    def _write_summary(self, result: WorkflowResult) -> None:
        try:
            result.properties = self._write_properties(result)
        except Exception as exc:                      # never lose a run over a summary
            self.log(f"    (could not write properties: {exc})")
        payload = result.summary()
        (self.rundir / "summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (self.rundir / "report.md").write_text(render_report(result, self.config),
                                               encoding="utf-8")

    def _cleanup(self) -> None:
        if self.config.get("output.keep_wavefunctions", False):
            return
        save = self.outdir
        if not save.is_dir():
            return
        patterns = ("wfc*.dat", "*.wfc*", "*.igk*")
        for path in [q for pattern in patterns for q in save.rglob(pattern)]:
            try:
                path.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------ report
def render_report(result: WorkflowResult, config) -> str:
    from ezcal.structures import structure_info

    lines = [f"# ezcal report - {result.task}", ""]
    lines.append(f"- run directory: `{result.rundir}`")
    lines.append(f"- status: {'OK' if result.ok else 'FAILED'}")
    lines.append(f"- wall time: {result.elapsed:.1f} s")
    lines.append(f"- engine: `{config.get('engine', 'qe')}`  "
                 f"scheduler: `{config.get('run.scheduler', 'local')}`  "
                 f"nproc: {config.get('run.nproc')}")
    lines.append("")

    if result.structure_initial is not None:
        info = structure_info(result.structure_initial)
        lines += ["## structure (input)", "",
                  f"- formula: **{info['formula']}**  ({info['natoms']} atoms)",
                  f"- space group: {info.get('spacegroup')} (#{info.get('spacegroup_number')})",
                  f"- a, b, c = {info['lattice']['a']:.4f}, {info['lattice']['b']:.4f}, "
                  f"{info['lattice']['c']:.4f} A",
                  f"- volume: {info['volume']:.3f} A^3", ""]

    lines += ["## parameters", "",
              f"- ecutwfc / ecutrho: {config.get('dft.ecutwfc')} / "
              f"{config.get('dft.ecutrho')} Ry",
              f"- functional: {config.get('dft.functional')}",
              f"- occupations: {config.get('dft.occupations')} "
              f"({config.get('dft.smearing')}, degauss={config.get('dft.degauss')} Ry)",
              f"- nspin: {config.get('dft.nspin')}", ""]

    magnetic = any(result.steps[n].magnetization is not None for n in result.order)
    header = ["step", "ok", "energy (eV)", "E/atom (eV)", "E_F (eV)", "gap (eV)",
              "max\\|F\\| (eV/A)", "P (GPa)"]
    if magnetic:
        header += ["M (uB/cell)", "abs M (uB/cell)"]
    header.append("time (s)")
    lines += ["## steps", "",
              "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)]
    for name in result.order:
        step = result.steps[name]

        def fmt(value, digits=4):
            return "-" if value is None else f"{value:.{digits}f}"

        row = [name, "yes" if step.ok else "no", fmt(step.energy, 6),
               fmt(step.energy_per_atom, 6), fmt(step.fermi_energy),
               fmt(step.band_gap, 3), fmt(step.max_force, 4), fmt(step.pressure, 2)]
        if magnetic:
            row += [fmt(step.magnetization, 3), fmt(step.abs_magnetization, 3)]
        row.append(fmt(step.walltime, 1))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    site_moments = next((result.steps[n].site_magnetization for n in result.order
                         if result.steps[n].site_magnetization), None)
    if site_moments:
        from ezcal.structures import site_labels

        names = (site_labels(result.structure_final)
                 if result.structure_final is not None else
                 [str(i + 1) for i in range(len(site_moments))])
        pairs = ", ".join(f"{name} {value:+.3f}"
                          for name, value in zip(names, site_moments))
        lines += [f"Magnetic moment per site (uB): {pairs}", ""]
    if result.order:
        gap_info = next((result.steps[n].data.get("gap_info") for n in reversed(result.order)
                         if result.steps[n].data.get("gap_info")), None)
        if gap_info:
            if gap_info.get("metal"):
                lines += ["The eigenvalue spectrum has no gap: **metallic**.", ""]
            else:
                kind = "direct" if gap_info.get("direct") else "indirect"
                lines += [f"Band gap: **{gap_info['gap']:.3f} eV** ({kind}), "
                          f"VBM {gap_info['vbm']:.3f} eV, CBM {gap_info['cbm']:.3f} eV. "
                          "Plots are referenced to the VBM.", ""]

    if result.plots:
        lines += ["## figures", ""]
        for path in result.plots:
            lines.append(f"- `{path}`")
        lines.append("")
    if result.exports:
        lines += ["## data", ""] + [f"- `{p}`" for p in result.exports] + [""]
    if result.messages:
        lines += ["## notes", ""] + [f"- {m}" for m in result.messages] + [""]
    return "\n".join(lines)


def render_properties(payload: Mapping[str, Any]) -> str:
    """The one-page answer to 'what came out of this calculation?'."""
    def fmt(value, digits=4, unit=""):
        if value is None:
            return "-"
        return f"{value:.{digits}f}{unit}"

    lines = [f"# {payload.get('formula') or 'properties'}", "",
             f"- run: `{payload.get('rundir')}`  (task `{payload.get('task')}`)", ""]

    electronic = payload.get("electronic") or {}
    if electronic:
        metal = electronic.get("metal")
        gap = electronic.get("band_gap_eV")
        lines += ["## Electronic structure", "", "| quantity | value |", "|---|---|",
                  f"| Fermi energy | {fmt(electronic.get('fermi_energy_eV'))} eV |"]
        if metal:
            lines.append("| band gap | **0** (metallic: a band crosses the Fermi level) |")
        else:
            kind = electronic.get("gap_kind")
            lines.append(f"| band gap | **{fmt(gap, 4)} eV**"
                         f"{f' ({kind})' if kind else ''} |")
            lines.append(f"| valence band maximum | {fmt(electronic.get('vbm_eV'))} eV |")
            lines.append(f"| conduction band minimum | {fmt(electronic.get('cbm_eV'))} eV |")
        lines += [f"| electrons in the cell | {fmt(electronic.get('nelec'), 1)} |",
                  f"| bands computed | {electronic.get('nbnd') or '-'} |",
                  f"| taken from | the `{electronic.get('source_step')}` step |", ""]

    magnetism = payload.get("magnetism") or {}
    if magnetism.get("nspin") == 2:
        lines += ["## Magnetism", "",
                  f"- total magnetization: **{fmt(magnetism.get('total_magnetization_uB'), 3)}** "
                  "μB/cell",
                  f"- absolute magnetization: **{fmt(magnetism.get('absolute_magnetization_uB'), 3)}** "
                  "μB/cell  (this is the meaningful one for an antiferromagnet)", ""]

    atoms = payload.get("atoms") or []
    if atoms:
        columns = [("index", "#", 0), ("label", "site", None), ("element", "element", None),
                   ("moment_sphere_uB", "moment (μB)", 3),
                   ("moment_lowdin_uB", "moment Löwdin (μB)", 3),
                   ("lowdin_charge_e", "Löwdin charge (e)", 3),
                   ("bader_charge_e", "Bader charge (e)", 3),
                   ("bader_volume_A3", "Bader volume (Å³)", 2)]
        present = [c for c in columns
                   if c[0] in {"index", "label", "element"}
                   or any(a.get(c[0]) is not None for a in atoms)]
        lines += ["## Per-atom", "",
                  "| " + " | ".join(c[1] for c in present) + " |",
                  "|" + "---|" * len(present)]
        for atom in atoms:
            cells = []
            for key, _, digits in present:
                value = atom.get(key)
                cells.append("-" if value is None else
                             (str(value) if digits is None else fmt(value, digits)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if any(a.get("bader_charge_e") is not None for a in atoms):
            lines += ["Bader charges are positive when electrons have been removed from the "
                      "atom.  They come from partitioning the density on the FFT grid, so "
                      "they carry a grid-size uncertainty of roughly 0.01-0.05 e.", ""]
    return "\n".join(lines)
