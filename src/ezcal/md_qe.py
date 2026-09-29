"""Quantum ESPRESSO による第一原理分子動力学 (``ezcal md-qe``)。

``ezcal md-mlip`` (material-mc + ASE calculator) と同じ後処理の流れに乗せた、
``pw.x`` ネイティブの Born-Oppenheimer MD である。力を 1 ステップずつ ASE から
呼ぶのではなく ``calculation='md'`` / ``'vc-md'`` の 1 回の ``pw.x`` 実行で済ませる
ので、電荷密度・波動関数の外挿がそのまま効き、qsub へ 1 ジョブで投げられる。

アンサンブルと pw.x の対応:

======== ============ ================================================
ensemble calculation  温度・圧力の制御
======== ============ ================================================
nve      md           ion_temperature='not_controlled' (初速のみ与える)
nvt      md           svr (既定) | berendsen | andersen | nose | rescaling ...
npt      vc-md        cell_dynamics='pr' + ion_temperature='rescaling'
nph      vc-md        cell_dynamics='pr'、初期温度だけ与えて以後は制御しない
======== ============ ================================================

実行後は ``md.out`` を解析し、md-mlip と同じ形の成果物を書き出す:
``energy_log.csv`` / ``energy_profile.png`` / ``plots/dynamics.*`` /
``structures/step_*.cif`` / ``combined.traj`` / ``initial.cif`` / ``final.cif`` /
``final_structure.cif`` / ``summary.json`` / ``report.md``。
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ezcal.md import DynamicsError, DynamicsResult, summarize_records

#: Rydberg 原子単位系の時間 (fs)。pw.x の dt はこの単位で与える
AU_TIME_FS = 0.04837768652
BOHR_ANG = 0.529177210903
RY_EV = 13.605693122994

ENSEMBLES = ("nve", "nvt", "npt", "nph")
#: calculation='md' で使える熱浴
MD_THERMOSTATS = ("svr", "berendsen", "andersen", "nose", "rescaling", "rescale-v",
                  "rescale-T", "reduce-T")
#: calculation='vc-md' (vcsmd.f90) が解釈する熱浴
VC_THERMOSTATS = ("rescaling", "nose")

MODE_NAME = "md-qe"
TITLE = "第一原理 MD (Quantum ESPRESSO pw.x)"


# ------------------------------------------------------------------ 設定
def _get(config, key: str, default: Any = None) -> Any:
    value = config.get(f"md_qe.{key}", None)
    return default if value is None else value


def md_settings(config) -> dict:
    """``md_qe:`` セクションを検証し、pw.x に渡す前の正規化済みの値にする。"""
    ensemble = str(_get(config, "ensemble", "nvt")).lower()
    if ensemble not in ENSEMBLES:
        raise DynamicsError(f"未知のアンサンブルです: {ensemble!r} "
                            f"(選べるのは {', '.join(ENSEMBLES)})")
    variable_cell = ensemble in {"npt", "nph"}
    timestep = float(_get(config, "timestep_fs", 2.0))
    if timestep <= 0:
        raise DynamicsError("timestep_fs は正の値にしてください")
    temperature = float(_get(config, "temperature_K", 300.0))
    init_t = _get(config, "init_temperature_K")
    init_t = temperature if init_t is None else float(init_t)

    thermostat = _get(config, "thermostat")
    if ensemble == "nvt":
        thermostat = str(thermostat or "svr")
        allowed = MD_THERMOSTATS
    elif ensemble == "npt":
        thermostat = str(thermostat or "rescaling")
        allowed = VC_THERMOSTATS
    else:                                   # nve / nph は温度を制御しない
        thermostat, allowed = "none", ("none",)
    if thermostat not in allowed:
        raise DynamicsError(
            f"{ensemble} で使える熱浴は {', '.join(allowed)} です (指定: {thermostat!r})")

    ttime = float(_get(config, "ttime_fs", 20.0))
    return {
        "ensemble": ensemble,
        "calculation": "vc-md" if variable_cell else "md",
        "thermostat": thermostat,
        "n_steps": int(_get(config, "steps", 50)),
        "timestep_fs": timestep,
        "temperature_K": temperature,
        "init_temperature_K": init_t,
        "ttime_fs": ttime,
        "nraise": max(1, int(round(ttime / timestep))),
        "tolp": float(_get(config, "tolp", 100.0)),
        "delta_t": float(_get(config, "delta_t", 1.0)),
        "fnosep_thz": _get(config, "fnosep_thz"),
        "pressure_gpa": float(_get(config, "pressure_gpa", 0.0)),
        "cell_dynamics": str(_get(config, "cell_dynamics", "pr")),
        "cell_dofree": str(_get(config, "cell_dofree", "all")),
        "wmass": _get(config, "wmass"),
        "cell_factor": _get(config, "cell_factor"),
        "pot_extrapolation": str(_get(config, "pot_extrapolation", "atomic")),
        "wfc_extrapolation": str(_get(config, "wfc_extrapolation", "none")),
        "conv_thr": _get(config, "conv_thr"),
        "seed": _get(config, "seed"),
        "npool": _get(config, "npool"),
        "save_interval": max(1, int(_get(config, "save_interval", 10))),
    }


def namelists(settings: Mapping[str, Any], natoms: int) -> dict[str, dict]:
    """``&CONTROL`` / ``&SYSTEM`` / ``&ELECTRONS`` への追記と ``&IONS`` / ``&CELL``。"""
    dt_au = settings["timestep_fs"] / AU_TIME_FS
    control = {"calculation": settings["calculation"], "dt": round(dt_au, 4),
               "nstep": settings["n_steps"], "tprnfor": True, "tstress": True}
    # 乱数の初速は対称性を壊す。対称化を残すと 2 ステップ目で checkallsym が止まり、
    # 反転対称があると start_therm が等価原子の速度を打ち消して 0 K から始まる
    system = {"nosym": True, "noinv": True}
    electrons: dict[str, Any] = {}
    if settings["conv_thr"] is not None:
        electrons["conv_thr"] = float(settings["conv_thr"])

    ions: dict[str, Any] = {
        "ion_dynamics": "beeman" if settings["calculation"] == "vc-md" else "verlet",
        "pot_extrapolation": settings["pot_extrapolation"],
        "wfc_extrapolation": settings["wfc_extrapolation"],
    }
    thermostat = settings["thermostat"]
    ensemble = settings["ensemble"]
    if ensemble == "nve":
        ions["ion_temperature"] = "not_controlled"
    elif ensemble == "nph":
        # vc-md の 'rescaling' は |T - tempw| > tolp のときだけ速度を揃える。
        # tolp を十分大きくすると、最初の熱化だけが効く (= 初期温度を与えた NPH)
        ions.update(ion_temperature="rescaling", tempw=settings["init_temperature_K"],
                    tolp=1.0e8, nraise=max(1, settings["n_steps"] + 1))
    else:
        ions["ion_temperature"] = thermostat
        ions["tempw"] = settings["temperature_K"]
        if thermostat in {"svr", "berendsen", "andersen", "rescale-v", "reduce-T"} or (
                thermostat == "rescaling" and ensemble == "npt"):
            ions["nraise"] = settings["nraise"]
        if thermostat == "rescaling":
            ions["tolp"] = settings["tolp"]
        if thermostat in {"rescale-T", "reduce-T"}:
            ions["delta_t"] = settings["delta_t"]
        if thermostat == "nose" and settings["fnosep_thz"] is not None:
            ions["fnosep"] = float(settings["fnosep_thz"])

    out = {"control": control, "system": system, "electrons": electrons, "ions": ions}
    if settings["calculation"] == "vc-md":
        cell: dict[str, Any] = {"cell_dynamics": settings["cell_dynamics"],
                                "press": settings["pressure_gpa"] * 10.0,   # GPa -> kbar
                                "cell_dofree": settings["cell_dofree"]}
        if settings["wmass"] is not None:
            cell["wmass"] = float(settings["wmass"])
        if settings["cell_factor"] is not None:
            out["system"]["cell_factor"] = float(settings["cell_factor"])
        out["cell"] = cell
    return out


def initial_velocities(atoms, temperature_K: float, seed=None) -> np.ndarray | None:
    """Maxwell-Boltzmann 分布の初速を pw.x の ``ATOMIC_VELOCITIES a.u.`` 単位で返す。

    重心運動は取り除き、温度はちょうど ``temperature_K`` に揃える
    (md-mlip の ``init_temperature_K`` と同じ振る舞い)。0 K なら ``None``。
    """
    if temperature_K <= 0 or len(atoms) < 2:
        return None
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

    work = atoms.copy()
    rng = np.random.default_rng(None if seed is None else int(seed))
    MaxwellBoltzmannDistribution(work, temperature_K=temperature_K, rng=rng,
                                 force_temp=True)
    Stationary(work, preserve_temperature=True)
    # ASE は 3N 自由度、pw.x は重心を除いた 3N-3 自由度で温度を定義する。
    # pw.x の出力する 0 ステップ目の温度が指定値と一致するよう揃えておく
    natoms = len(work)
    work.set_velocities(work.get_velocities()
                        * np.sqrt((3 * natoms - 3) / (3 * natoms)))
    # ASE の速度 [Å / ASE 時間] -> [Å/fs] -> [bohr / Rydberg 時間]
    return work.get_velocities() * units.fs / BOHR_ANG * AU_TIME_FS


def render_input(builder, extra_blocks: Mapping[str, Mapping[str, Any]],
                 velocities: np.ndarray | None) -> str:
    """:class:`PwInput` のカードに ``&IONS`` / ``&CELL`` / ``ATOMIC_VELOCITIES`` を足す。"""
    from ezcal.engines.qe import inputs as qein
    from ezcal.structures import site_labels

    blocks = [qein.namelist("control", builder.control()),
              qein.namelist("system", builder.system()),
              qein.namelist("electrons", builder.electrons()),
              qein.namelist("ions", extra_blocks["ions"])]
    if "cell" in extra_blocks:
        blocks.append(qein.namelist("cell", extra_blocks["cell"]))
    blocks += [builder.card_species(), builder.card_cell(), builder.card_positions(),
               builder.card_kpoints()]
    hubbard = builder.card_hubbard()
    if hubbard:
        blocks.append(hubbard)
    if velocities is not None:
        lines = ["ATOMIC_VELOCITIES a.u."]
        for label, (vx, vy, vz) in zip(site_labels(builder.structure), velocities):
            lines.append(f"  {label:<4s} {vx:18.12e} {vy:18.12e} {vz:18.12e}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


# --------------------------------------------------------------- 出力の解析
_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?|\*+)"
_MD_BLOCK = re.compile(r"Entering Dynamics:\s+iteration\s*=\s*(\d+)")
_VC_BLOCK = re.compile(r"Entering Dynamics;\s+it\s*=\s*(\d+)")
_MD_EKIN = re.compile(r"kinetic energy \(Ekin\)\s*=\s*" + _FLOAT + r"\s*Ry")
_MD_TEMP = re.compile(r"temperature\s*=\s*" + _FLOAT + r"\s*K")
_MD_CONST = re.compile(r"Ekin \+ Etot \(const\)\s*=\s*" + _FLOAT + r"\s*Ry")
_VC_LINE = re.compile(r"Ekin\s*=\s*" + _FLOAT + r"\s*Ry\s+T\s*=\s*" + _FLOAT
                      + r"\s*K\s+Etot\s*=\s*" + _FLOAT)
_SCF_ITER = re.compile(r"convergence has been achieved in\s+(\d+)\s+iterations")


def _num(text: str | None) -> float | None:
    if text is None or text.startswith("*"):
        return None                              # Fortran の書式あふれ
    return float(text.replace("D", "E").replace("d", "E"))


def parse_dynamics_blocks(text: str) -> list[dict]:
    """``Entering Dynamics`` ブロックごとの運動エネルギー・温度・保存量 (eV, K)。"""
    marks = [(m.start(), "md") for m in _MD_BLOCK.finditer(text)]
    marks += [(m.start(), "vc") for m in _VC_BLOCK.finditer(text)]
    marks.sort()
    blocks: list[dict] = []
    for index, (start, kind) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(text)
        chunk = text[start:end]
        row: dict[str, float | None] = {}
        if kind == "md":
            ekin, temp, const = (_MD_EKIN.search(chunk), _MD_TEMP.search(chunk),
                                 _MD_CONST.search(chunk))
            row["e_kin_eV"] = None if ekin is None else _num(ekin.group(1)) * RY_EV
            row["temperature_K"] = None if temp is None else _num(temp.group(1))
            if const is not None:
                row["e_const_eV"] = _num(const.group(1)) * RY_EV
        else:
            # vcsmd.f90 の T は「これまでの平均運動エネルギー」から (N+1) 自由度で
            # 出した値で、瞬間温度ではない。瞬間温度は build_records で Ekin から出す
            row["variable_cell"] = True
            match = _VC_LINE.search(chunk)
            if match:
                ekin = _num(match.group(1))
                row["e_kin_eV"] = None if ekin is None else ekin * RY_EV
                row["temperature_run_avg_K"] = _num(match.group(2))
                etot = _num(match.group(3))
                if etot is not None:
                    row["e_const_eV"] = etot * RY_EV
        blocks.append(row)
    return blocks


def _unwrapped_msd(frames) -> list[float]:
    """スケール座標を最小像で繋いで展開し、初期位置からの MSD (Å^2) を出す。"""
    if not frames:
        return []
    previous = frames[0].get_scaled_positions(wrap=False)
    unwrapped = previous.copy()
    origin = frames[0].positions.copy()
    out = [0.0]
    for atoms in frames[1:]:
        current = atoms.get_scaled_positions(wrap=False)
        delta = current - previous
        delta -= np.round(delta)
        unwrapped = unwrapped + delta
        previous = current
        cart = unwrapped @ atoms.cell.array
        out.append(float(np.mean(np.sum((cart - origin) ** 2, axis=1))))
    return out


def build_records(frames, blocks: Sequence[Mapping], timestep_fs: float, phase: str,
                  scf_iterations: Sequence[int] = ()) -> list[dict]:
    """SCF 1 回 (= 1 フレーム) につき 1 行の記録を作る。

    pw.x は i 回目の SCF のあとに i 番目の ``Entering Dynamics`` ブロックを書く。
    そのブロックの運動エネルギーと温度は、i 回目の SCF の座標での速度に対応する。
    """
    from ase import units

    msd = _unwrapped_msd(frames)
    records: list[dict] = []
    for index, atoms in enumerate(frames):
        row: dict[str, Any] = {"step": index, "phase": phase,
                               "time_fs": index * timestep_fs}
        results = getattr(atoms.calc, "results", {}) if atoms.calc else {}
        energy = results.get("energy")
        row["energy_eV"] = None if energy is None else float(energy)
        block = blocks[index] if index < len(blocks) else {}
        ekin = block.get("e_kin_eV")
        row["e_kin_eV"] = ekin
        row["e_total_eV"] = (None if ekin is None or energy is None
                             else float(energy) + float(ekin))
        row["temperature_K"] = block.get("temperature_K")
        if block.get("variable_cell") and ekin is not None and len(atoms) > 1:
            row["temperature_K"] = 2.0 * float(ekin) / (3 * (len(atoms) - 1) * units.kB)
        row["e_const_eV"] = block.get("e_const_eV")
        row["volume_A3"] = float(atoms.get_volume())
        stress = results.get("stress")
        if stress is not None:
            row["pressure_GPa"] = float(-np.mean(np.asarray(stress)[:3]) / units.GPa)
        forces = results.get("forces")
        if forces is not None:
            row["fmax_eV_A"] = float(np.linalg.norm(np.asarray(forces), axis=1).max())
        row["msd_A2"] = msd[index]
        if index < len(scf_iterations):
            row["scf_iterations"] = int(scf_iterations[index])
        records.append(row)
    return records


def read_md_output(path: str | Path, timestep_fs: float, phase: str):
    """``md.out`` -> (ASE のフレーム列, 記録の行)。"""
    from ase.io import read

    path = Path(path)
    if not path.is_file():
        return [], []
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        frames = read(str(path), index=":", format="espresso-out")
    except Exception:
        frames = []
    if not isinstance(frames, list):
        frames = [frames]
    blocks = parse_dynamics_blocks(text)
    iterations = [int(m.group(1)) for m in _SCF_ITER.finditer(text)]
    return frames, build_records(frames, blocks, timestep_fs, phase, iterations)


CSV_COLUMNS = ("step", "phase", "time_fs", "energy_eV", "e_kin_eV", "e_total_eV",
               "temperature_K", "volume_A3", "pressure_GPa", "msd_A2", "fmax_eV_A",
               "e_const_eV", "scf_iterations")


def write_energy_log(records: Sequence[Mapping], path: str | Path) -> Path:
    """md-mlip と同じ ``energy_log.csv`` (空欄は値なし)。"""
    path = Path(path)
    columns = [c for c in CSV_COLUMNS if any(r.get(c) is not None for r in records)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in records:
            writer.writerow(["" if row.get(c) is None else row[c] for c in columns])
    return path


# ------------------------------------------------------------------ 実行本体
def _prepare_structure(config, structure, log: Callable[[str], None]):
    """スーパーセル化した pymatgen 構造と、それと同じ座標系の ASE Atoms を返す。

    pw.x はセルの向きを問わないので、md-mlip と違ってセルの三角化は行わない
    (初速を ASE で作るため、両者の座標系を一致させておく必要がある)。
    """
    from ezcal.structures import to_ase
    from ezcal.workflows import Workflow

    # 反強磁性の副格子ラベル (Fe1/Fe2) は繰り返す前に振る。こうすると磁気秩序が
    # そのままスーパーセル全体に複製される
    structure = Workflow._apply_sublattices(config, structure, log)
    supercell = _get(config, "supercell")
    if supercell:
        structure = structure * [int(n) for n in supercell]
        log(f"    スーパーセル {tuple(int(n) for n in supercell)} -> {len(structure)} 原子")
    return structure, to_ase(structure)


def describe(config, kmesh: Sequence[int] | None = None) -> str:
    functional = config.get("dft.input_dft") or config.get("dft.functional", "pbe")
    text = (f"Quantum ESPRESSO pw.x ({str(functional).upper()}, "
            f"ecutwfc={config.get('dft.ecutwfc')} Ry")
    if kmesh:
        text += f", k={'x'.join(str(int(k)) for k in kmesh)}"
    return text + ")"


def run_md_qe(config, structure, rundir: str | Path, collect_only: bool = False,
              log: Callable[[str], None] = print) -> DynamicsResult:
    """pw.x で MD を 1 回実行し (または既存の ``md.out`` を読み)、後処理まで済ませる。"""
    from ezcal.engines.qe import inputs as qein
    from ezcal.engines.qe.engine import QEEngine
    from ezcal.scheduler import Command, Stage, get_scheduler

    settings = md_settings(config)
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)
    structure, atoms = _prepare_structure(config, structure, log)
    phase = f"md-{settings['ensemble']}"

    engine = QEEngine(config, get_scheduler(config))
    infile, outfile = rundir / "md.in", rundir / "md.out"
    result = DynamicsResult(mode=MODE_NAME, workdir=rundir, natoms=len(atoms),
                            formula=atoms.get_chemical_formula())

    if collect_only:
        if not outfile.is_file():
            raise DynamicsError(f"{outfile} がありません (--collect は実行済みの "
                                "ディレクトリを指定してください)")
        log(f"    既存の {outfile.name} を解析します (pw.x は起動しません)")
        try:                         # report の calculator 欄を実行時と同じ表記にする
            engine.prepare(structure)
            kmesh = engine._kmesh(config, structure)
        except Exception:
            kmesh = None
    else:
        problems = engine.check()
        if problems and not config.get("run.dry_run") and \
                str(config.get("run.scheduler", "local")) == "local":
            raise DynamicsError("; ".join(problems))
        pseudos = engine.prepare(structure)
        kmesh = engine._kmesh(config, structure)
        blocks = namelists(settings, len(atoms))
        velocities = None
        if settings["calculation"] == "md" and settings["init_temperature_K"] > 0:
            # 初速は ezcal が与える: シードで再現でき、NVE でも初期温度を指定できる
            velocities = initial_velocities(atoms, settings["init_temperature_K"],
                                            settings["seed"])
            if velocities is not None:
                blocks["ions"]["ion_velocities"] = "from_input"
        outdir = (rundir / "tmp").resolve()
        builder = qein.PwInput(
            structure=structure, pseudos=pseudos, config=config,
            calculation=settings["calculation"], prefix=engine.prefix,
            outdir=str(outdir), pseudo_dir=str(engine.pseudo_dir()), kmesh=kmesh,
            extra={k: v for k, v in blocks.items() if k in {"control", "system", "electrons"}},
        )
        infile.write_text(render_input(builder, blocks, velocities), encoding="utf-8")
        structure.to(filename=str(rundir / "initial.cif"))
        args = ["-in", infile.name]
        if settings["npool"]:
            args += ["-nk", str(int(settings["npool"]))]
        stage = Stage(name="md", workdir=rundir,
                      commands=[Command(engine.executable("pw"), args, stdout=outfile,
                                        label="pw.x md")])
        result.messages += [f"warning: {w}" for w in engine.warnings]

    result.calculator = describe(config, kmesh)
    result.parameters = {"engine": "qe", **{k: v for k, v in settings.items()
                                            if v is not None}}
    if kmesh:
        result.parameters["kmesh"] = list(kmesh)
    result.parameters["ecutwfc_Ry"] = config.get("dft.ecutwfc")
    result.parameters["ecutrho_Ry"] = config.get("dft.ecutrho")
    log(f"    calculator: {result.calculator}")
    log(f"    {TITLE}: " + ", ".join(
        f"{k}={settings[k]}" for k in ("ensemble", "thermostat", "n_steps", "timestep_fs",
                                       "temperature_K")))

    start = time.time()
    if not collect_only:
        job = engine.scheduler.execute(stage)
        result.walltime = job.elapsed
        if job.script:
            result.exports.append(Path(job.script))
        if job.submitted_only:
            result.ok = True
            result.messages.append(
                f"ジョブ {job.job_id} を投入しました。終了後に --collect で後処理できます"
                if job.job_id else
                "ドライラン: md.in とコマンドを書き出しました (pw.x は起動していません)")
            _write_outputs(result, config, [], [])
            return result
        if not job.ok:
            result.messages += [line for line in job.log[-2:] if line]
            result.messages.append(f"pw.x が失敗しました (終了コード {job.returncode})。"
                                   f"{outfile} を確認してください")

    frames, records = read_md_output(outfile, settings["timestep_fs"], phase)
    if collect_only:
        result.walltime = _pw_walltime(outfile) or (time.time() - start)
    if not frames:
        result.messages.append("md.out から MD のフレームを読めませんでした")
        _write_outputs(result, config, frames, records)
        raise DynamicsError(result.messages[-1])

    result.energy_initial = records[0].get("energy_eV")
    result.energy_final = records[-1].get("energy_eV")
    text = outfile.read_text(encoding="utf-8", errors="replace")
    finished = "End of molecular dynamics calculation" in text or \
        "JOB DONE" in text
    result.ok = finished and len(records) >= min(settings["n_steps"], 1)
    if not finished:
        result.messages.append(f"pw.x の MD が最後まで終わっていません "
                               f"({len(records)}/{settings['n_steps']} ステップ)")
    _write_outputs(result, config, frames, records)
    if result.ok and not config.get("output.keep_wavefunctions", False):
        _remove_wavefunctions(rundir / "tmp")
    return result


def _pw_walltime(path: Path) -> float | None:
    from ezcal.engines.qe import outputs as qeout

    try:
        return qeout.parse_pw_text(path).walltime
    except Exception:
        return None


def _remove_wavefunctions(tmp: Path) -> None:
    if not tmp.is_dir():
        return
    for pattern in ("*.wfc*", "*.oldwfc*", "*.old2wfc*", "wfc*.dat", "*.igk*",
                    "*.save/wfc*.dat"):
        for path in tmp.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass


# --------------------------------------------------------------- 成果物の出力
def _write_outputs(result: DynamicsResult, config, frames, records) -> None:
    import json

    from ezcal import plotting
    from ezcal.structures import from_ase

    rundir = result.workdir
    result.n_records = len(records)
    result.averages = summarize_records(records)
    pressures = [r["pressure_GPa"] for r in records if r.get("pressure_GPa") is not None]
    if pressures:
        tail = pressures[int(len(pressures) * 0.5):] or pressures
        result.averages["mean_pressure_GPa"] = float(sum(tail) / len(tail))
    drift = _energy_drift(records)
    if drift is not None:
        result.averages["drift_e_const_meV_per_atom_ps"] = drift / max(1, result.natoms)

    if records:
        result.exports.append(write_energy_log(records, rundir / "energy_log.csv"))
    if frames:
        _write_structures(result, config, frames)
        try:
            final = from_ase(frames[-1])
            final.to(filename=str(rundir / "final_structure.cif"))
            result.exports.append(rundir / "final_structure.cif")
        except Exception as exc:
            result.messages.append(f"final_structure.cif を書けませんでした: {exc}")

    backends = plotting.resolve_backends(config.get("output.plot"))
    if records:
        try:
            _energy_profile(records, rundir / "energy_profile.png",
                            f"{result.formula} {result.mode}")
        except Exception as exc:
            result.messages.append(f"energy_profile.png を描けませんでした: {exc}")
    if records and backends:
        try:
            result.plots += plotting.plot_dynamics(
                records, rundir / "plots", backends,
                dpi=int(config.get("output.dpi", 200)),
                title=f"{result.formula} {result.mode} ({result.parameters.get('ensemble')})")
        except Exception as exc:
            result.messages.append(f"作図に失敗しました: {exc}")

    (rundir / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, default=str), encoding="utf-8")
    (rundir / "report.md").write_text(render_report(result), encoding="utf-8")


def _energy_drift(records: Sequence[Mapping]) -> float | None:
    """保存量 (Ekin + Etot) の線形ドリフト (meV/ps)。NVE の品質確認に使う。"""
    pairs = [(r["time_fs"], r["e_const_eV"]) for r in records
             if r.get("e_const_eV") is not None and r.get("time_fs") is not None]
    if len(pairs) < 3:
        return None
    t, e = np.array(pairs, dtype=float).T
    slope = np.polyfit(t, e, 1)[0]                 # eV/fs
    return float(slope * 1.0e6)                    # meV/ps


def _write_structures(result: DynamicsResult, config, frames) -> None:
    from ase.io import write

    rundir = result.workdir
    folder = rundir / "structures"
    folder.mkdir(exist_ok=True)
    for stale in folder.glob("step_*.cif"):          # 同じディレクトリでの再実行に備える
        stale.unlink()
    interval = max(1, int(_get(config, "save_interval", 10)))
    for index, atoms in enumerate(frames):
        if index % interval == 0 or index == len(frames) - 1:
            write(str(folder / f"step_{index:07d}.cif"), atoms)
    try:
        write(str(folder / "md_0000000.traj"), frames)
        if bool(_get(config, "combine_traj", True)):
            write(str(rundir / "combined.traj"), frames)
            result.exports.append(rundir / "combined.traj")
        write(str(rundir / "final.cif"), frames[-1])
        if not (rundir / "initial.cif").is_file():
            write(str(rundir / "initial.cif"), frames[0])
    except Exception as exc:
        result.messages.append(f"traj を書けませんでした: {exc}")


def _energy_profile(records: Sequence[Mapping], path: Path, title: str) -> Path:
    """material-mc の ``energy_profile.png`` に相当する 1 枚図 (E_pot と E_tot)。"""
    from ezcal.plotting import _mpl

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.0, 3.6), layout="constrained")
    steps = [r["step"] for r in records]
    ax.plot(steps, [r.get("energy_eV") for r in records], ".-", ms=3, lw=1.0,
            color="#1f4e9c", label="E$_{pot}$")
    if any(r.get("e_total_eV") is not None for r in records):
        ax.plot(steps, [np.nan if r.get("e_total_eV") is None else r["e_total_eV"]
                        for r in records], ".-", ms=3, lw=1.0, color="#b0392c",
                label="E$_{pot}$ + E$_{kin}$")
    ax.set_xlabel("MD step")
    ax.set_ylabel("energy (eV)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ レポート
def render_report(result: DynamicsResult) -> str:
    def fmt(value, digits=4, unit=""):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}{unit}"
        return f"{value}{unit}"

    params = result.parameters
    lines = [f"# {result.formula} — {TITLE}", "",
             f"- モード: `{result.mode}` (pw.x calculation='{params.get('calculation')}'、"
             f"ensemble={params.get('ensemble')}、thermostat={params.get('thermostat')})",
             f"- 実行ディレクトリ: `{result.workdir}`",
             f"- calculator: `{result.calculator}`",
             f"- 原子数: {result.natoms}",
             f"- 所要時間: {result.walltime:.1f} s",
             f"- 記録点数: {result.n_records}", ""]

    lines += ["## 設定", "", "| パラメータ | 値 |", "|---|---|"]
    for key, value in params.items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")

    lines += ["## エネルギー", "", "| 量 | 値 |", "|---|---|",
              f"| 初期ポテンシャルエネルギー | {fmt(result.energy_initial, 6, ' eV')} |",
              f"| 最終ポテンシャルエネルギー | {fmt(result.energy_final, 6, ' eV')} |"]
    if result.energy_initial is not None and result.energy_final is not None:
        delta = result.energy_final - result.energy_initial
        lines.append(f"| 変化 | {fmt(delta, 6, ' eV')} "
                     f"({fmt(delta / max(1, result.natoms) * 1000, 3, ' meV/原子')}) |")
    for key, label in (("mean_temperature_K", "平均温度 (後半)"),
                       ("mean_volume_A3", "平均体積 (後半)"),
                       ("mean_pressure_GPa", "平均圧力 (後半, GPa)"),
                       ("mean_e_total_eV", "平均全エネルギー (後半)"),
                       ("mean_msd_A2", "平均 MSD (後半)"),
                       ("drift_e_const_meV_per_atom_ps",
                        "保存量ドリフト (meV/原子/ps)")):
        if result.averages.get(key) is not None:
            lines.append(f"| {label} | {fmt(result.averages[key], 4)} |")
    lines.append("")

    lines += ["## 出力", ""]
    for path in result.plots:
        lines.append(f"- 図 `{Path(path).name}`")
    for path in result.exports:
        lines.append(f"- データ `{Path(path).name}`")
    lines += ["- 記録 `energy_log.csv` / `energy_profile.png`",
              "- 構造 `structures/` (CIF スナップショットと MD traj)",
              "- pw.x の入出力 `md.in` / `md.out`", ""]
    for message in result.messages:
        lines.append(f"> {message}")
    return "\n".join(lines) + "\n"
