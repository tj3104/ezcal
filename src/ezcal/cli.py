"""The ``ezcal`` command line interface."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from ezcal import __version__
from ezcal.config import Config, flatten, load_config, parse_set_options
from ezcal.engines.base import EngineError
from ezcal.scheduler import SchedulerError
from ezcal.structures import StructureError

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="ezcal - one command first-principles calculations "
         "(Quantum ESPRESSO by default).",
)
config_app = typer.Typer(no_args_is_help=True, help="inspect and create qe_config.yaml")
pseudo_app = typer.Typer(no_args_is_help=True, help="pseudopotential library helpers")
vasp_app = typer.Typer(no_args_is_help=True,
                       help="VASP helpers: mock-vasp registry of recorded runs")
bench_app = typer.Typer(no_args_is_help=True,
                        help="run the built-in benchmark suite and compare with "
                             "experiment and the Materials Project")
app.add_typer(config_app, name="config")
app.add_typer(pseudo_app, name="pseudo")
app.add_typer(vasp_app, name="vasp")
app.add_typer(bench_app, name="bench")

console = Console()
TASKS = ["scf", "relax", "vc-relax", "nscf", "bands", "dos", "charge", "auto"]


# --------------------------------------------------------------- utilities
def _fail(message: str, code: int = 1) -> None:
    console.print(f"[bold red]error[/]: {message}")
    raise typer.Exit(code)


def _parse_mapping(value: Optional[str], cast=float, allow_list: bool = False) -> dict:
    """``"Fe=0.5,O=0.1"`` -> ``{"Fe": 0.5, "O": 0.1}``.

    With ``allow_list`` a "/" separates the values of magnetic sublattices:
    ``"Fe=0.6/-0.6"`` -> ``{"Fe": [0.6, -0.6]}``.
    """
    if not value:
        return {}
    out: dict[str, Any] = {}
    for chunk in value.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk and ":" not in chunk:
            raise typer.BadParameter(f"expected element=value, got {chunk!r}")
        key, _, raw = chunk.replace(":", "=").partition("=")
        raw = raw.strip()
        if allow_list and "/" in raw:
            out[key.strip()] = [cast(part) for part in raw.split("/") if part.strip()]
        else:
            out[key.strip()] = cast(raw)
    return out


def _parse_afm(value: Optional[str]) -> dict:
    """``"Fe"`` -> ``{"Fe": None}``;  ``"Fe=1.0,Ni=2.0"`` -> magnitudes."""
    out: dict[str, Any] = {}
    for chunk in (value or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, _, raw = chunk.partition("=")
            try:
                out[key.strip()] = abs(float(raw))
            except ValueError as exc:
                raise typer.BadParameter(
                    f"--afm expects an element or element=magnitude, got {chunk!r}") from exc
        else:
            out[chunk] = None
    return out


def _build_config(**kw) -> Config:
    cfg = load_config(kw.get("config"))
    overrides: dict[str, Any] = {}

    def put(key, value):
        if value is not None:
            overrides[key] = value

    put("engine", kw.get("engine"))
    put("dft.ecutwfc", kw.get("ecutwfc"))
    put("dft.ecutrho", kw.get("ecutrho"))
    put("dft.kspacing", kw.get("kspacing"))
    put("dft.functional", kw.get("functional"))
    put("dft.smearing", kw.get("smearing"))
    put("dft.degauss", kw.get("degauss"))
    put("dft.occupations", kw.get("occupations"))
    put("dft.conv_thr", kw.get("conv_thr"))
    put("dft.nbnd", kw.get("nbnd"))
    put("dft.vdw_corr", kw.get("vdw"))
    put("dft.input_dft", kw.get("input_dft"))
    put("qe.pseudo_dir", kw.get("pseudo_dir"))
    put("run.nproc", kw.get("nproc"))
    put("output.dpi", kw.get("dpi"))
    put("bands.line_density", kw.get("line_density"))
    put("bands.emin", kw.get("emin"))
    put("bands.emax", kw.get("emax"))

    if kw.get("kmesh"):
        overrides["dft.kmesh"] = [int(v) for v in kw["kmesh"]]
    if kw.get("spin"):
        overrides["dft.nspin"] = 2
    magmom = _parse_mapping(kw.get("magmom"), allow_list=True)
    scalars = {k: v for k, v in magmom.items() if not isinstance(v, list)}
    sublattices = {k: v for k, v in magmom.items() if isinstance(v, list)}
    for element, magnitude in _parse_afm(kw.get("afm")).items():
        size = magnitude if magnitude is not None else abs(scalars.pop(element, 0.0)) or 0.5
        sublattices.setdefault(element, [size, -size])
    if scalars:
        overrides["dft.starting_magnetization"] = scalars
    if sublattices:
        overrides["dft.magnetic_sublattices"] = sublattices
    if magmom or sublattices:
        overrides["dft.nspin"] = 2
    hubbard = _parse_mapping(kw.get("hubbard_u"))
    if hubbard:
        overrides["dft.hubbard_u"] = hubbard
    pins = _parse_mapping(kw.get("pseudo"), cast=str)
    if pins:
        overrides["qe.pseudo_map"] = pins

    scheduler = kw.get("scheduler")
    if kw.get("qsub"):
        scheduler = "qsub"
    put("run.scheduler", scheduler)
    if kw.get("dry_run"):
        overrides["run.dry_run"] = True
    if kw.get("qsub_wait"):
        overrides["run.qsub.wait"] = True
    put("run.qsub.queue", kw.get("queue"))
    put("run.qsub.walltime", kw.get("walltime"))
    put("run.qsub.nodes", kw.get("nodes"))
    put("run.qsub.ppn", kw.get("ppn"))
    put("run.qsub.script", kw.get("qsub_script"))

    if kw.get("plot"):
        overrides["output.plot"] = [p.strip() for p in str(kw["plot"]).split(",")]
    if kw.get("keep_wfc"):
        overrides["output.keep_wavefunctions"] = True
    if kw.get("charge_kinds"):
        overrides["charge.kinds"] = [k.strip() for k in str(kw["charge_kinds"]).split(",")
                                     if k.strip()]
    if kw.get("no_bader"):
        overrides["charge.bader"] = False

    for dotted, value in parse_set_options(kw.get("set_options")).items():
        overrides[dotted] = value

    cfg.apply_overrides(overrides)
    return cfg


def _load_structure(source: str, primitive: bool, mp_api_key: str | None):
    from ezcal.structures import read_structure, standardize

    try:
        structure = read_structure(source, api_key=mp_api_key or os.environ.get("MP_API_KEY"))
    except Exception as exc:
        _fail(str(exc))
    if primitive:
        try:
            structure = standardize(structure, primitive=True)
        except Exception as exc:
            console.print(f"[yellow]warning[/]: could not standardise the cell ({exc})")
    return structure


def _rundir(cfg: Config, structure, task: str, outdir: Optional[str],
            name: Optional[str]) -> Path:
    root = Path(outdir) if outdir else Path(str(cfg.get("output.dir", "ezcal_out")))
    label = name or f"{structure.composition.reduced_formula}_{task}"
    return (root / label).resolve()


def _print_header(cfg: Config, structure, task: str, rundir: Path) -> None:
    from ezcal.structures import structure_info

    info = structure_info(structure)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    table.add_row("task", task)
    table.add_row("formula", f"{info['formula']}  ({info['natoms']} atoms)")
    table.add_row("space group", f"{info.get('spacegroup')} (#{info.get('spacegroup_number')})")
    table.add_row("engine", str(cfg.get("engine")))
    table.add_row("scheduler", f"{cfg.get('run.scheduler')} (np={cfg.get('run.nproc')})")
    table.add_row("run dir", str(rundir))
    console.print(table)


def _print_result(result) -> None:
    magnetic = any(result.steps[n].magnetization is not None for n in result.order)
    table = Table(title="results", header_style="bold")
    columns = ["step", "ok", "E (eV)", "E/atom (eV)", "E_F (eV)", "gap (eV)",
               "max|F|", "P (GPa)"]
    if magnetic:
        columns += ["M (uB)", "|M| (uB)"]
    columns.append("t (s)")
    for column in columns:
        table.add_column(column, justify="right")
    for name in result.order:
        step = result.steps[name]

        def fmt(value, digits=4):
            return "-" if value is None else f"{value:.{digits}f}"

        row = [name, "[green]yes[/]" if step.ok else "[red]no[/]",
               fmt(step.energy, 6), fmt(step.energy_per_atom, 6),
               fmt(step.fermi_energy), fmt(step.band_gap, 3),
               fmt(step.max_force), fmt(step.pressure, 2)]
        if magnetic:
            row += [fmt(step.magnetization, 3), fmt(step.abs_magnetization, 3)]
        row.append(fmt(step.walltime, 1))
        table.add_row(*row)
    console.print(table)
    for path in result.plots:
        console.print(f"  figure  [green]{path}[/]")
    for path in result.exports:
        console.print(f"  data    [green]{path}[/]")
    console.print(f"  report  [green]{result.rundir / 'report.md'}[/]")
    console.print(f"  summary [green]{result.rundir / 'summary.json'}[/]")
    for message in result.messages:
        console.print(f"  [yellow]note[/]: {message}")


# ------------------------------------------------------------ command body
def _run_task(task: str, structure_arg: str, kw: dict) -> None:
    from ezcal.workflows import Workflow

    cfg = _build_config(**kw)
    primitive = kw.get("primitive", True)
    if cfg.get("dft.magnetic_sublattices") and primitive:
        # reducing to the primitive nuclear cell would throw away the very
        # sites the magnetic ordering needs
        primitive = False
        console.print("[yellow]note[/]: magnetic sublattices requested - keeping the cell "
                      "as given (--as-is) so the ordering fits")
    structure = _load_structure(structure_arg, primitive, kw.get("mp_api_key"))

    if task == "auto" and kw.get("relax") is False:
        kw["skip"] = ",".join(filter(None, [kw.get("skip"), "vc-relax,relax"]))
    if task == "auto" and kw.get("fixed_cell"):
        from ezcal import workflows

        workflows.CHAINS["auto"] = ("relax", "scf", "nscf", "dos", "bands")
    if task == "auto" and kw.get("with_charge"):
        from ezcal import workflows

        workflows.CHAINS["auto"] = (*workflows.CHAINS["auto"], "charge")

    skip = [s.strip() for s in (kw.get("skip") or "").split(",") if s.strip()]
    rundir = _rundir(cfg, structure, task, kw.get("outdir"), kw.get("name"))
    _print_header(cfg, structure, task, rundir)

    try:
        workflow = Workflow(cfg, structure, rundir, label=kw.get("name") or "",
                            log=lambda msg: console.print(f"[dim]{msg}[/]"))
    except (StructureError, EngineError, SchedulerError) as exc:
        _fail(str(exc))
    steps = workflow.plan(task, skip=skip, only=bool(kw.get("only")))
    console.print(f"[dim]    plan: {' -> '.join(steps) or '(nothing to do)'}[/]")

    try:
        result = workflow.run(task, skip=skip, only=bool(kw.get("only")))
    except (StructureError, EngineError, SchedulerError) as exc:
        _fail(str(exc))
    cfg.save(rundir / "qe_config.used.yaml")     # resolved values, for reproducibility

    _print_result(result)
    if not result.ok:
        raise typer.Exit(1)


def _make_command(task: str):
    def command(
        structure: str = typer.Argument(..., metavar="STRUCTURE",
                                        help="CIF/POSCAR/xyz file or a Materials Project id"),
        # --- DFT ---------------------------------------------------------
        ecutwfc: Optional[float] = typer.Option(None, help="wavefunction cutoff (Ry)"),
        ecutrho: Optional[float] = typer.Option(None, help="charge density cutoff (Ry)"),
        kmesh: Optional[tuple[int, int, int]] = typer.Option(
            None, "--kmesh", help="Monkhorst-Pack mesh, e.g. --kmesh 8 8 8"),
        kspacing: Optional[float] = typer.Option(
            None, help="reciprocal spacing in 1/Ang used when --kmesh is absent"),
        functional: Optional[str] = typer.Option(
            None, "--functional", "-f", help="pbe | pbesol | pz (picks the pseudopotentials)"),
        input_dft: Optional[str] = typer.Option(None, help="override QE's input_dft"),
        occupations: Optional[str] = typer.Option(
            None, help="smearing | fixed | tetrahedra"),
        smearing: Optional[str] = typer.Option(None, help="mv | gaussian | mp | fd"),
        degauss: Optional[float] = typer.Option(None, help="smearing width (Ry)"),
        conv_thr: Optional[float] = typer.Option(None, "--conv-thr", help="SCF threshold (Ry)"),
        nbnd: Optional[int] = typer.Option(None, help="number of bands"),
        spin: bool = typer.Option(False, "--spin", "--magnetic",
                                  help="collinear spin polarised (nspin=2)"),
        magmom: Optional[str] = typer.Option(
            None, help="starting magnetisation, e.g. --magmom Fe=0.6,O=0; "
                       "use '/' for sublattices, e.g. --magmom Fe=0.6/-0.6"),
        afm: Optional[str] = typer.Option(
            None, "--afm",
            help="antiferromagnetic: split an element into +/- sublattices, "
                 "e.g. --afm Cr or --afm Ni=2.0"),
        hubbard_u: Optional[str] = typer.Option(
            None, "--hubbard-u", help="DFT+U values, e.g. --hubbard-u Fe=4.0"),
        vdw: Optional[str] = typer.Option(None, help="vdw_corr, e.g. grimme-d3"),
        # --- pseudopotentials -------------------------------------------
        pseudo_dir: Optional[str] = typer.Option(None, help="local UPF directory"),
        pseudo: Optional[str] = typer.Option(
            None, help="pin a UPF file, e.g. --pseudo Fe=Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"),
        # --- bands / dos -------------------------------------------------
        line_density: Optional[float] = typer.Option(
            None, "--line-density", help="band path k-points per 1/Ang"),
        emin: Optional[float] = typer.Option(None, help="plot window minimum, eV from E_F"),
        emax: Optional[float] = typer.Option(None, help="plot window maximum, eV from E_F"),
        # --- execution ---------------------------------------------------
        nproc: Optional[int] = typer.Option(None, "--np", "-n", help="MPI ranks"),
        scheduler: Optional[str] = typer.Option(None, help="local | qsub"),
        qsub: bool = typer.Option(False, "--qsub", help="shortcut for --scheduler qsub"),
        qsub_wait: bool = typer.Option(False, "--qsub-wait",
                                       help="block until the queued job finishes"),
        qsub_script: Optional[str] = typer.Option(None, "--qsub-script",
                                                  help="job script template (run_qe.sh)"),
        queue: Optional[str] = typer.Option(None, help="queue name for qsub"),
        walltime: Optional[str] = typer.Option(None, help="walltime for qsub"),
        nodes: Optional[int] = typer.Option(None, help="nodes for qsub"),
        ppn: Optional[int] = typer.Option(None, help="processes per node for qsub"),
        dry_run: bool = typer.Option(False, "--dry-run",
                                     help="write the inputs and job scripts, run nothing"),
        # --- workflow ----------------------------------------------------
        only: bool = typer.Option(False, "--only",
                                  help="run only this step, skip its prerequisites"),
        skip: Optional[str] = typer.Option(None, help="comma separated steps to skip"),
        relax: bool = typer.Option(True, "--relax/--no-relax",
                                   help="(auto) relax the structure first"),
        fixed_cell: bool = typer.Option(False, "--fixed-cell",
                                        help="(auto) relax ions only, keep the cell"),
        with_charge: bool = typer.Option(False, "--charge",
                                         help="(auto) also dump the charge density and "
                                              "compute atomic charges"),
        primitive: bool = typer.Option(True, "--primitive/--as-is",
                                       help="standardise to the primitive cell first"),
        # --- output ------------------------------------------------------
        outdir: Optional[str] = typer.Option(None, "--outdir", "-o", help="results root"),
        name: Optional[str] = typer.Option(None, help="run directory name"),
        plot: Optional[str] = typer.Option(
            None, help="matplotlib | plotly | both | none"),
        dpi: Optional[int] = typer.Option(None, help="raster resolution"),
        keep_wfc: bool = typer.Option(False, "--keep-wfc", help="keep the wavefunction files"),
        charge_kinds: Optional[str] = typer.Option(
            None, "--charge-kinds",
            help="what pp.x should dump: density,spin,ae_valence,ae_total,potential"),
        no_bader: bool = typer.Option(False, "--no-bader",
                                      help="skip the Bader partition of the density"),
        # --- misc --------------------------------------------------------
        engine: Optional[str] = typer.Option(None, help="qe | vasp | mlip"),
        config: Optional[str] = typer.Option(None, "--config", "-c",
                                             help="path to qe_config.yaml"),
        set_options: Optional[list[str]] = typer.Option(
            None, "--set", help="override any config key, e.g. --set dft.mixing_beta=0.2"),
        mp_api_key: Optional[str] = typer.Option(
            None, "--mp-api-key", envvar="MP_API_KEY", help="Materials Project API key"),
    ) -> None:
        _run_task(task, structure, dict(locals()))

    command.__name__ = task.replace("-", "_")
    command.__doc__ = {
        "scf": "Self-consistent field calculation.",
        "relax": "Relax the atomic positions.",
        "vc-relax": "Relax the atomic positions and the cell.",
        "nscf": "Non self-consistent calculation on a dense mesh (runs scf first).",
        "bands": "Band structure along an automatic high-symmetry path (runs scf first).",
        "dos": "Total and projected density of states (runs scf and nscf first).",
        "charge": "Charge density cubes plus Loewdin and Bader atomic charges "
                  "(runs scf and dos first).",
        "auto": "Everything from a bare structure: vc-relax -> scf -> nscf -> dos -> bands.",
    }[task]
    return command


for _task in TASKS:
    app.command(_task)(_make_command(_task))


# ------------------------------------------------------------ config group
@config_app.command("init")
def config_init(
    path: str = typer.Option("qe_config.yaml", "--path", "-p", help="where to write it"),
    force: bool = typer.Option(False, "--force", help="overwrite an existing file"),
    user: bool = typer.Option(False, "--user", help="write to ~/.config/ezcal/"),
) -> None:
    """Write a fully commented qe_config.yaml you can edit."""
    from ezcal.config import PACKAGE_DEFAULT, USER_CONFIG_PATH

    target = USER_CONFIG_PATH if user else Path(path)
    if target.exists() and not force:
        _fail(f"{target} already exists (use --force)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(PACKAGE_DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"wrote [green]{target}[/]")


@config_app.command("show")
def config_show(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    flat: bool = typer.Option(False, "--flat", help="one dotted key per line"),
) -> None:
    """Print the effective configuration and where it came from."""
    cfg = load_config(config)
    console.print("[dim]# sources (later wins):[/]")
    for source in cfg.sources:
        console.print(f"[dim]#   {source}[/]")
    if flat:
        for key, value in flatten(cfg.data):
            console.print(f"{key} = {value}")
    else:
        console.print(cfg.to_yaml())


# ------------------------------------------------------------ pseudo group
@pseudo_app.command("list")
def pseudo_list(
    element: str = typer.Argument(..., help="element symbol, e.g. Fe"),
    functional: str = typer.Option("pbe", "--functional", "-f"),
    relativistic: bool = typer.Option(False, "--relativistic"),
) -> None:
    """Show the pseudopotentials ezcal knows about for one element."""
    from ezcal.pseudo import candidates, rank_candidates

    files = rank_candidates(element, candidates(element, functional, relativistic))
    if not files:
        _fail(f"no {functional} pseudopotential indexed for {element}")
    for i, name in enumerate(files):
        marker = "[green]*[/]" if i == 0 else " "
        console.print(f" {marker} {name}")


@pseudo_app.command("fetch")
def pseudo_fetch(
    elements: list[str] = typer.Argument(..., help="element symbols"),
    functional: str = typer.Option("pbe", "--functional", "-f"),
    pseudo_dir: Optional[str] = typer.Option(None, "--pseudo-dir"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Download pseudopotentials into the local library and report their cutoffs."""
    from ezcal.pseudo import PseudoManager

    cfg = load_config(config)
    manager = PseudoManager(
        pseudo_dir=pseudo_dir or cfg.get("qe.pseudo_dir", "~/.ezcal/pseudo"),
        functional=functional,
        preference=cfg.get("qe.pseudo_preference", ["kjpaw", "rrkjus"]),
    )
    table = Table(header_style="bold")
    for column in ("element", "file", "type", "Z_val", "ecutwfc (Ry)", "ecutrho (Ry)"):
        table.add_column(column)
    infos = []
    for element in elements:
        try:
            info = manager.resolve(element)
        except Exception as exc:
            _fail(str(exc))
        infos.append(info)
        table.add_row(info.element, info.filename, info.pseudo_type,
                      f"{info.z_valence:g}",
                      "-" if info.ecutwfc is None else f"{info.ecutwfc:g}",
                      "-" if info.ecutrho is None else f"{info.ecutrho:g}")
    console.print(table)
    wfc, rho = PseudoManager.suggest_cutoffs(infos)
    console.print(f"suggested for this set: ecutwfc={wfc} Ry, ecutrho={rho} Ry")
    console.print(f"library: [green]{manager.pseudo_dir}[/]")


# ------------------------------------------------------------- bench group
@bench_app.command("list")
def bench_list(
    which: str = typer.Option("all", "--set", "-s", help="metals | oxides | all"),
) -> None:
    """Show the systems in the benchmark suite and their reference values."""
    from ezcal.benchmark import load_suite

    try:
        entries = load_suite(which)
    except Exception as exc:
        _fail(str(exc))
    table = Table(header_style="bold")
    for column in ("system", "formula", "prototype", "atoms", "a exp (A)", "c exp (A)",
                   "gap exp (eV)", "B0 exp (GPa)", "m exp (uB)", "spin"):
        table.add_column(column, justify="right")
    for entry in entries:
        reference = entry.reference

        def show(key, digits=3):
            value = reference.get(key)
            return "-" if value is None else f"{float(value):.{digits}f}"

        table.add_row(entry.name, entry.formula, entry.prototype,
                      str(entry.natoms or "?"), show("a", 4), show("c", 4),
                      show("gap", 2), show("b0", 1), show("magmom", 2),
                      "AFM" if entry.afm else ("FM" if entry.nspin == 2 else "-"))
    console.print(table)
    console.print(f"[dim]{len(entries)} systems[/]")


@bench_app.command("run")
def bench_run(
    which: str = typer.Option("all", "--set", "-s", help="metals | oxides | all"),
    outdir: str = typer.Option("03_qe_bench", "--outdir", "-o", help="results root"),
    only: Optional[str] = typer.Option(None, "--only",
                                       help="comma separated system names to run"),
    limit: Optional[int] = typer.Option(None, "--limit", help="stop after N systems"),
    task: Optional[str] = typer.Option(None, "--task", help="override the task (default auto)"),
    nproc: Optional[int] = typer.Option(None, "--np", "-n", help="MPI ranks"),
    engine: str = typer.Option("qe", "--engine", help="qe | vasp | mlip"),
    eos: bool = typer.Option(True, "--eos/--no-eos",
                             help="also fit an equation of state for the bulk modulus"),
    eos_points: int = typer.Option(5, "--eos-points", help="volumes in the E(V) fit"),
    eos_range: float = typer.Option(0.04, "--eos-range", help="max strain in the E(V) fit"),
    resume: bool = typer.Option(True, "--resume/--restart",
                                help="skip systems that already have results"),
    plot: str = typer.Option("matplotlib", "--plot", help="matplotlib | plotly | both | none"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    set_options: Optional[list[str]] = typer.Option(None, "--set-config",
                                                    help="override any config key"),
) -> None:
    """Run the benchmark suite (this is the long one - hours for the full set)."""
    import numpy as np

    from ezcal.benchmark import load_suite, run_suite

    cfg = load_config(config)
    overrides: dict[str, Any] = {"engine": engine, "output.plot": [plot]}
    if nproc:
        overrides["run.nproc"] = nproc
    overrides.update(parse_set_options(set_options))
    cfg.apply_overrides(overrides)

    try:
        entries = load_suite(which)
    except Exception as exc:
        _fail(str(exc))
    if only:
        wanted = {name.strip() for name in only.split(",") if name.strip()}
        entries = [e for e in entries if e.name in wanted]
        missing = wanted - {e.name for e in entries}
        if missing:
            _fail(f"unknown system(s): {', '.join(sorted(missing))}")
    if task:
        for entry in entries:
            entry.task = task
    if limit:
        entries = entries[:limit]
    if not entries:
        _fail("nothing to run")

    strains = np.linspace(-abs(eos_range), abs(eos_range), max(4, eos_points)).tolist()
    root = Path(outdir).resolve()
    console.print(f"[bold]{len(entries)}[/] systems -> [green]{root}[/]  "
                  f"(engine {engine}, np {cfg.get('run.nproc')}, "
                  f"{'with' if eos else 'no'} EOS)")

    records = run_suite(entries, cfg, root, eos=eos, resume=resume, strains=strains,
                        log=lambda msg: console.print(f"[dim]{msg}[/]"))

    ok = sum(1 for r in records if r.get("ok"))
    console.print(f"\n[bold]{ok}/{len(records)}[/] finished. "
                  f"Now run: [green]ezcal bench report {root}[/]")
    if ok < len(records):
        raise typer.Exit(1)


@bench_app.command("report")
def bench_report(
    rundir: str = typer.Argument("03_qe_bench", help="benchmark results directory"),
    mp_api_key: Optional[str] = typer.Option(None, "--mp-api-key", envvar="MP_API_KEY",
                                             help="also compare with the Materials Project"),
    plot: str = typer.Option("both", "--plot", help="matplotlib | plotly | both | none"),
    dpi: int = typer.Option(200, "--dpi"),
) -> None:
    """Compare a finished benchmark run with experiment and the Materials Project."""
    from ezcal.benchmark import compare, fetch_mp, load_results, summarise, write_report

    root = Path(rundir)
    try:
        records = load_results(root)
    except Exception as exc:
        _fail(str(exc))

    mp_data: dict = {}
    if mp_api_key:
        console.print("[dim]looking up Materials Project entries...[/]")
        try:
            mp_data = fetch_mp(records, mp_api_key,
                               log=lambda msg: console.print(f"[dim]{msg}[/]"))
        except Exception as exc:
            console.print(f"[yellow]Materials Project lookup skipped[/]: {exc}")
    else:
        console.print("[dim]no MP API key given - comparing with experiment only[/]")

    rows = compare(records, mp_data)
    files = write_report(records, rows, root, backends=[plot], dpi=dpi)

    for against, title in (("exp", "experiment"), ("mp", "Materials Project")):
        summary = summarise(rows, against)
        if not summary:
            continue
        table = Table(title=f"mean errors vs {title}", header_style="bold")
        for column in ("quantity", "set", "n", "ME", "MAE", "MRE %", "MARE %"):
            table.add_column(column, justify="right")
        for entry in summary:
            table.add_row(f"{entry['label']} ({entry['unit'] or '-'})", entry["group"],
                          str(entry["n"]), f"{entry['me']:+.4g}", f"{entry['mae']:.4g}",
                          f"{entry['mre']:+.2f}", f"{entry['mare']:.2f}")
        console.print(table)

    console.print(f"  report  [green]{files['report']}[/]")
    console.print(f"  data    [green]{files['csv']}[/]")
    for path in files.get("plots", []):
        console.print(f"  figure  [green]{path}[/]")


# -------------------------------------------------------------- vasp group
def _vasp_engine(config_path: Optional[str], registry: Optional[str] = None):
    from ezcal.engines import get_engine

    cfg = load_config(config_path)
    if registry:
        cfg.set("vasp.mock.registry", registry)
    return cfg, get_engine("vasp", cfg)


@vasp_app.command("record")
def vasp_record(
    rundir: str = typer.Argument(..., help="a finished VASP run directory"),
    name: str = typer.Option(..., "--name", "-n",
                             help="label for the entry, e.g. si-scf/calc-000"),
    registry: Optional[str] = typer.Option(None, "--registry", "-r",
                                           help="registry directory (default: vasp.mock.registry)"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Add a finished VASP run to the mock registry so mock-vasp can replay it.

    Run this where VASP actually works; copy the registry to machines that have
    no licence and they can re-run the same inputs through ezcal.
    """
    _, engine = _vasp_engine(config, registry)
    try:
        path = engine.record(Path(rundir), name)
    except Exception as exc:
        _fail(str(exc))
    console.print(f"recorded [green]{rundir}[/] as [green]{path}[/]")


@vasp_app.command("registry")
def vasp_registry(
    registry: Optional[str] = typer.Option(None, "--registry", "-r"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """List the calculations mock-vasp can replay."""
    from aiida_vasp.utils.mock_code import VaspMockRegistry

    _, engine = _vasp_engine(config, registry)
    base = engine.registry_path()
    if base is None:
        _fail("no registry configured (set vasp.mock.registry or pass --registry)")
    reg = VaspMockRegistry(str(base))
    if not reg.reg_name:
        console.print(f"[yellow]no entries found under[/] {base}")
        raise typer.Exit(0)
    table = Table(header_style="bold")
    table.add_column("entry")
    table.add_column("hash")
    table.add_column("path")
    for name, digest in sorted(reg.reg_name.items()):
        table.add_row(name, digest, str(reg.reg_hash[digest]))
    console.print(table)


@vasp_app.command("hash")
def vasp_hash(
    folder: str = typer.Argument(..., help="directory holding INCAR/KPOINTS/POSCAR"),
) -> None:
    """Print the mock-vasp hash of an input folder (useful to debug a miss)."""
    from aiida_vasp.utils.mock_code import VaspMockRegistry

    path = Path(folder)
    if not (path / "INCAR").is_file():
        _fail(f"{path} has no INCAR")
    console.print(VaspMockRegistry.compute_hash(path))


# -------------------------------------------------------------- misc group
@app.command("info")
def info(
    structure: str = typer.Argument(..., metavar="STRUCTURE"),
    primitive: bool = typer.Option(True, "--primitive/--as-is"),
    kspacing: float = typer.Option(0.25, help="spacing used for the suggested mesh"),
    mp_api_key: Optional[str] = typer.Option(None, "--mp-api-key", envvar="MP_API_KEY"),
    band_path_only: bool = typer.Option(False, "--band-path", help="also print the k-path"),
) -> None:
    """Inspect a structure: symmetry, suggested k-mesh, pseudopotentials, cutoffs."""
    from ezcal.pseudo import PseudoManager
    from ezcal.structures import auto_kmesh, band_path, structure_info

    struct = _load_structure(structure, primitive, mp_api_key)
    data = structure_info(struct)
    console.print_json(json.dumps(data, indent=2))
    console.print(f"suggested k-mesh (kspacing={kspacing}): {auto_kmesh(struct, kspacing)}")
    try:
        manager = PseudoManager()
        infos = manager.resolve_all([str(el) for el in struct.composition.elements])
        wfc, rho = PseudoManager.suggest_cutoffs(infos.values())
        for element, item in infos.items():
            console.print(f"  {element:<3s} {item.filename}  "
                          f"({item.pseudo_type}, Z={item.z_valence:g})")
        console.print(f"suggested cutoffs: ecutwfc={wfc} Ry  ecutrho={rho} Ry")
    except Exception as exc:
        console.print(f"[yellow]pseudopotentials unavailable[/]: {exc}")
    if band_path_only:
        path = band_path(struct)
        console.print(f"k-path ({path.nkpt} points): "
                      + " -> ".join(dict.fromkeys(l for _, l in path.labels if l)))


@app.command("mp")
def mp_get(
    material_id: str = typer.Argument(..., help="e.g. mp-149"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="output CIF path"),
    mp_api_key: Optional[str] = typer.Option(None, "--mp-api-key", envvar="MP_API_KEY"),
) -> None:
    """Download a structure from the Materials Project and save it as CIF."""
    from ezcal.structures import from_materials_project

    try:
        structure = from_materials_project(material_id, mp_api_key)
    except Exception as exc:
        _fail(str(exc))
    target = Path(out or f"{material_id}.cif")
    structure.to(filename=str(target))
    console.print(f"{structure.composition.reduced_formula} -> [green]{target}[/]")


@app.command("engines")
def list_engines(config: Optional[str] = typer.Option(None, "--config", "-c")) -> None:
    """Show the available engines and whether they can run here."""
    from ezcal.engines import available, get_engine

    cfg = load_config(config)
    table = Table(header_style="bold")
    for column in ("engine", "tasks", "status"):
        table.add_column(column)
    for name in ("qe", "vasp", "mlip"):
        engine = get_engine(name, cfg)
        problems = engine.check()
        status = "[green]ready[/]" if not problems else f"[yellow]{problems[0]}[/]"
        table.add_row(name, ", ".join(engine.supported), status)
    console.print(table)
    console.print(f"[dim]aliases: {', '.join(available())}[/]")


@app.command("plot")
def replot(
    rundir: str = typer.Argument(..., help="a previous ezcal run directory"),
    plot: str = typer.Option("both", help="matplotlib | plotly | both"),
    emin: float = typer.Option(-10.0, help="energy window minimum (eV from E_F)"),
    emax: float = typer.Option(10.0, help="energy window maximum (eV from E_F)"),
    dpi: int = typer.Option(200),
    title: Optional[str] = typer.Option(None, help="figure title"),
) -> None:
    """Redraw the figures of a finished run (no recalculation)."""
    import numpy as np

    from ezcal import plotting
    from ezcal.engines.base import CalcResult

    root = Path(rundir)
    raw_path = root / "raw_data.json"
    if not raw_path.is_file():
        _fail(f"{raw_path} not found - rerun the calculation, or point at the run directory")
    payload = json.loads(raw_path.read_text())
    backends = plotting.resolve_backends([p.strip() for p in plot.split(",")])
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    label = title or root.name
    fermi = payload.get("fermi")
    zero = payload.get("zero", "F")

    bands_result = dos_result = None
    if payload.get("bands"):
        bands_result = CalcResult(task="bands", engine="qe", workdir=root, ok=True)
        bands_result.data["eigenvalues"] = np.asarray(payload["bands"]["eigenvalues"])
        bands_result.data["kpath"] = payload["bands"].get("kpath", {})
    if payload.get("dos"):
        dos_result = CalcResult(task="dos", engine="qe", workdir=root, ok=True)
        dos_result.data["dos"] = {k: (np.asarray(v) if isinstance(v, list) else v)
                                  for k, v in payload["dos"].items()}
        if payload.get("pdos"):
            dos_result.data["pdos"] = {
                key: ({k: np.asarray(v) for k, v in value.items()}
                      if isinstance(value, dict)
                      else (np.asarray(value) if isinstance(value, list) else value))
                for key, value in payload["pdos"].items()
            }

    written: list[Path] = []
    if bands_result is not None:
        written += plotting.plot_bands(bands_result, plots_dir, backends, fermi=fermi,
                                       emin=emin, emax=emax, dpi=dpi, zero=zero,
                                       title=f"{label} band structure")
    if dos_result is not None:
        written += plotting.plot_dos(dos_result, plots_dir, backends, fermi=fermi,
                                     emin=emin, emax=emax, dpi=dpi, zero=zero,
                                     title=f"{label} density of states")
    if bands_result is not None and dos_result is not None:
        written += plotting.plot_bands_dos(bands_result, dos_result, plots_dir, backends,
                                           fermi=fermi, emin=emin, emax=emax, dpi=dpi,
                                           zero=zero, title=label)
    if not written:
        _fail("nothing to plot: this run has no band or DOS data")
    for path in written:
        console.print(f"[green]{path}[/]")


@app.command("version")
def version() -> None:
    """Print the ezcal version."""
    console.print(f"ezcal {__version__}  (python {sys.version.split()[0]})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
