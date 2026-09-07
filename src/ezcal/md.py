"""分子動力学とモンテカルロ (material-mc ラッパ)。

``ezcal md`` / ``ezcal mc`` の実体。計算そのものは material-mc
(:class:`mc.MaterialMonteCalro`) が行い、ここが受け持つのは ezcal 側の作法に
合わせる部分だけである:

* 構造の読み込みとスーパーセル化 (CIF/POSCAR/mp-id は他のタスクと共通)
* :mod:`ezcal.calculators` による ASE calculator の解決
  (SevenNet などの MLIP、あるいはユーザーの Python スクリプト)
* ``qe_config.yaml`` の ``md:`` セクションから各モードの引数を組み立てる
* 実行後の後処理 — エネルギー・温度・体積・MSD の作図 (matplotlib / plotly)、
  ``summary.json`` / ``report.md`` / ``final_structure.cif`` の書き出し

material-mc のチュートリアルにある計算はすべて :data:`MODES` から選べる。
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class DynamicsError(RuntimeError):
    pass


# --------------------------------------------------------------- モード定義
@dataclass(frozen=True)
class ModeSpec:
    """1 つの実行モード = material-mc の ``run_*`` メソッド 1 つ。"""

    name: str
    method: str
    title: str
    kind: str                       # md | mc | cycle | event
    aliases: tuple[str, ...] = ()
    note: str = ""

    @property
    def cyclic(self) -> bool:
        return self.kind == "cycle"


_MODE_LIST = (
    ModeSpec("md", "run_md", "MD 単独 (NVE/NVT/NPT/NPH)", "md",
             note="MC を挟まない純粋な MD。ensemble で N P V T E のどれを固定するか選ぶ"),
    ModeSpec("mcmc", "run_mcmc", "格子モンテカルロ", "mc",
             note="原子位置は固定し、元素スワップを Metropolis 判定でサンプリング"),
    ModeSpec("mcmd", "run_mcmd", "MC + MD ハイブリッド", "cycle",
             note="1 サイクル = MD md_steps 回 -> MC mc_steps 回"),
    ModeSpec("mcrelax", "run_mcrelax", "MC + 構造緩和", "relax",
             note="1 サイクル = MC mc_steps 回 -> 構造緩和 (0 K 的な安定配置探索)"),
    ModeSpec("kmc", "run_kmc_simple", "動的 MC (近接スワップ)", "mc",
             aliases=("kmc-simple",),
             note="スワップ相手をカットオフ内の近接原子に限定した MC"),
    ModeSpec("kmc-voronoi", "run_kmc_voronoi", "動的 MC (ボロノイ提案 + MH 補正)", "mc",
             aliases=("kmcvoronoi",),
             note="ボロノイ多面体を作る原子を候補にし、距離のガウス重みで抽選する"),
    ModeSpec("kmcmd", "run_kmcmd", "動的 MC (近接スワップ) + MD", "cycle",
             note="拡散加速 (Tavenner et al. 2023)。活性化エネルギー不要"),
    ModeSpec("kmc-voronoi-md", "run_kmc_voronoi_md", "ボロノイ動的 MC + MD", "cycle",
             aliases=("kmcvoronoimd",),
             note="カットオフではなく幾何で候補を決めるので、MD で歪んでも再調整が要らない"),
    ModeSpec("event-kmc", "run_event_kmc", "イベント定義型 kMC (CI-NEB + BKL)", "event",
             aliases=("eventkmc", "kmc-event"),
             note="空孔ホップの障壁を NEB で求め、実時間・拡散係数・イオン伝導度を出す"),
)

#: モード名 (別名込み) -> :class:`ModeSpec`
MODES: dict[str, ModeSpec] = {}
for _spec in _MODE_LIST:
    MODES[_spec.name] = _spec
    for _alias in _spec.aliases:
        MODES[_alias] = _spec

MODE_NAMES = tuple(spec.name for spec in _MODE_LIST)


def get_mode(name: str) -> ModeSpec:
    key = str(name).lower().replace("_", "-")
    if key not in MODES:
        raise DynamicsError(
            f"未知のモードです: {name!r}  (選べるのは {', '.join(MODE_NAMES)})")
    return MODES[key]


# ----------------------------------------------------------------- 結果保持
@dataclass
class DynamicsResult:
    mode: str
    workdir: Path
    ok: bool = False
    walltime: float = 0.0
    natoms: int = 0
    formula: str = ""
    calculator: str = ""
    parameters: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    averages: dict = field(default_factory=dict)
    event: dict = field(default_factory=dict)      # event-kMC 固有の戻り値
    plots: list[Path] = field(default_factory=list)
    exports: list[Path] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    energy_initial: float | None = None
    energy_final: float | None = None
    n_records: int = 0

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "workdir": str(self.workdir),
            "formula": self.formula,
            "natoms": self.natoms,
            "calculator": self.calculator,
            "walltime_s": self.walltime,
            "parameters": self.parameters,
            "energy_initial_eV": self.energy_initial,
            "energy_final_eV": self.energy_final,
            "n_records": self.n_records,
            "acceptance": self.stats,
            "averages": self.averages,
            "event_kmc": self.event or None,
            "plots": [str(p) for p in self.plots],
            "exports": [str(p) for p in self.exports],
            "messages": self.messages,
        }


# ------------------------------------------------------------- 引数の組み立て
def _get(config, key: str, default: Any = None) -> Any:
    value = config.get(f"md.{key}", None)
    return default if value is None else value


def sanitize_cell(atoms, tol: float = 1e-6):
    """セル行列を厳密な三角行列に整える (ASE の NPT 積分器の要求)。

    ``ase.md.npt.NPT`` は ``m[1,0] == m[2,0] == m[2,1] == 0.0`` を厳密な等号で
    確かめる。ところが CIF の読み書きや spglib による標準化を経たセルは、格子
    定数と角度から再構成される過程で 1e-16 程度の残差を非対角成分に持つ。
    material-mc の ``ensure_triangular_cell`` は ``np.allclose`` で判定するため
    この残差を「三角である」と見なして通してしまい、その後 ASE 側の厳密判定で
    ``NotImplementedError`` になる。ここで残差をゼロに丸め、三角でないセルは
    標準姿勢へ剛体回転しておく (スケール座標は不変なので物理は変わらない)。
    """
    import numpy as np

    cell = np.array(atoms.cell.array, dtype=float)
    if not cell.any():
        return atoms
    for form in (np.triu, np.tril):
        snapped = form(cell)
        if np.allclose(cell, snapped, atol=tol):
            if not np.array_equal(cell, snapped):
                atoms.set_cell(snapped, scale_atoms=False)
            return atoms

    from ase.cell import Cell

    scaled = atoms.get_scaled_positions()
    velocities = atoms.get_velocities()
    rebuilt = np.array(Cell.fromcellpar(atoms.cell.cellpar()).array, dtype=float)
    rebuilt[np.abs(rebuilt) < tol] = 0.0
    atoms.set_cell(rebuilt)
    atoms.set_scaled_positions(scaled)
    if velocities is not None and np.any(velocities):
        atoms.set_velocities(velocities @ np.linalg.inv(cell) @ rebuilt)
    return atoms


def _normalize_species(value) -> list | None:
    """``["Fe","Pt"]`` はそのまま、``[["Fe","Pt"],["Li","O"]]`` はタプルに直す。"""
    if not value:
        return None
    out: list = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.append(tuple(str(x) for x in item))
        else:
            out.append(str(item))
    return out


def _md_kwargs(config) -> dict:
    mask = _get(config, "npt_mask")
    return {
        "ensemble": str(_get(config, "ensemble", "nvt")).lower(),
        "timestep_fs": float(_get(config, "timestep_fs", 1.0)),
        "ttime_fs": float(_get(config, "ttime_fs", 25.0)),
        "pfactor_gpa_fs2": (None if _get(config, "pfactor_gpa_fs2") is None
                            else float(_get(config, "pfactor_gpa_fs2"))),
        "pressure_gpa": float(_get(config, "pressure_gpa", 0.0)),
        "npt_mask": None if mask is None else tuple(int(m) for m in mask),
        "tchain": int(_get(config, "tchain", 3)),
        "md_record_interval": int(_get(config, "record_interval", 1)),
    }


def build_arguments(config, mode: ModeSpec, atoms) -> dict:
    """``run_*`` に渡す引数を設定から組み立てる。"""
    if mode.kind == "md":
        kwargs = _md_kwargs(config)
        kwargs["n_steps"] = int(_get(config, "steps", 500))
        init_t = _get(config, "init_temperature_K")
        if init_t is not None:
            kwargs["init_temperature_K"] = float(init_t)
        if _get(config, "reinit_velocities", False):
            kwargs["reinit_velocities"] = True
        return kwargs

    if mode.kind == "mc":
        kwargs: dict[str, Any] = {"n_steps": int(_get(config, "steps", 500))}
        if mode.name == "kmc":
            kwargs["neighbor_cutoff"] = float(_get(config, "neighbor_cutoff", 3.5))
        if mode.name == "kmc-voronoi":
            r0 = _get(config, "r0")
            kwargs["r0"] = None if r0 is None else float(r0)
            kwargs["voronoi_cutoff"] = float(_get(config, "voronoi_cutoff", 6.0) or 6.0)
        return kwargs

    if mode.kind == "cycle":
        kwargs = _md_kwargs(config)
        kwargs.update(n_cycles=int(_get(config, "cycles", 5)),
                      mc_steps=int(_get(config, "mc_steps", 20)),
                      md_steps=int(_get(config, "md_steps", 100)))
        if mode.name == "kmcmd":
            kwargs["neighbor_cutoff"] = float(_get(config, "neighbor_cutoff", 3.5))
        if mode.name == "kmc-voronoi-md":
            r0 = _get(config, "r0")
            kwargs["r0"] = None if r0 is None else float(r0)
            kwargs["voronoi_cutoff"] = float(_get(config, "voronoi_cutoff", 0.0) or 0.0)
        return kwargs

    if mode.kind == "relax":
        return {"n_cycles": int(_get(config, "cycles", 5)),
                "mc_steps": int(_get(config, "mc_steps", 20)),
                "fmax": float(_get(config, "fmax", 0.05)),
                "max_itr": int(_get(config, "max_itr", 200)),
                "optimizer": str(_get(config, "optimizer", "FIRE")),
                "relax_record_interval": int(_get(config, "record_interval", 1))}

    # event-kMC
    mobile = _get(config, "mobile_species")
    if not mobile:
        raise DynamicsError(
            "event-kmc には可動元素の指定が要ります: --mobile-species Li "
            "(または md.mobile_species)")
    return {"n_steps": int(_get(config, "steps", 100)),
            "mobile_species": str(mobile),
            "nu0_hz": float(_get(config, "nu0_hz", 1.0e13)),
            "barrier_method": str(_get(config, "barrier_method", "neb")),
            "n_images": int(_get(config, "n_images", 5)),
            "neb_fmax": float(_get(config, "neb_fmax", 0.05)),
            "neb_steps": int(_get(config, "neb_steps", 300)),
            "env_cutoff": float(_get(config, "env_cutoff", 6.0)),
            "charge": float(_get(config, "charge", 1.0)),
            "energy_interval": int(_get(config, "energy_interval", 50)),
            "hop_cutoff": (None if _get(config, "hop_cutoff") is None
                           else float(_get(config, "hop_cutoff")))}


def make_vacancies(atoms, config, mode: ModeSpec, log: Callable[[str], None]):
    """event-kMC 用に空孔を作り、その座標を返す (他モードでは何もしない)。"""
    import numpy as np

    if mode.kind != "event":
        return atoms, None
    indices = _get(config, "vacancy_indices")
    if indices is None:
        count = int(_get(config, "vacancy_count", 1) or 0)
        if count <= 0:
            return atoms, None
        mobile = str(_get(config, "mobile_species") or "")
        candidates = [i for i, s in enumerate(atoms.get_chemical_symbols())
                      if s == mobile]
        if len(candidates) < count:
            raise DynamicsError(
                f"{mobile} が {len(candidates)} 個しかないため空孔を {count} 個作れません")
        rng = np.random.default_rng(_get(config, "seed"))
        indices = sorted(rng.choice(candidates, size=count, replace=False).tolist())
    indices = sorted(int(i) for i in indices)
    positions = [atoms.positions[i].copy() for i in indices]
    for index in reversed(indices):
        del atoms[index]
    log(f"    空孔を {len(indices)} 個導入しました (削除した原子: {indices})")
    return atoms, positions


# -------------------------------------------------------------- 記録の読み出し
def read_energy_log(path: str | Path) -> list[dict]:
    """material-mc の ``energy_log.csv`` を読む (空欄は None)。"""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None or value in ("", None):
                    continue
                if key == "phase":
                    row[key] = value
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            if row:
                rows.append(row)
    return rows


def average_tail(rows: Sequence[Mapping], key: str, fraction: float = 0.5) -> float | None:
    """後半 ``fraction`` の平均 (平衡到達後の代表値として使う)。"""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    start = int(len(values) * (1.0 - fraction))
    tail = values[start:] or values
    return float(sum(tail) / len(tail))


def summarize_records(rows: Sequence[Mapping]) -> dict:
    out: dict[str, Any] = {}
    for key, label in (("energy_eV", "energy_eV"),
                       ("temperature_K", "temperature_K"),
                       ("e_total_eV", "e_total_eV"),
                       ("volume_A3", "volume_A3"),
                       ("msd_A2", "msd_A2")):
        value = average_tail(rows, key)
        if value is not None:
            out[f"mean_{label}"] = value
    phases = []
    for row in rows:
        phase = row.get("phase")
        if phase and phase not in phases:
            phases.append(phase)
    if phases:
        out["phases"] = phases
    return out


# ------------------------------------------------------------------- 実行本体
def run_dynamics(config, structure, rundir: str | Path, mode: str | None = None,
                 log: Callable[[str], None] = print) -> DynamicsResult:
    """MD / MC を 1 回実行し、後処理まで済ませる。"""
    from ezcal import calculators
    from ezcal.calculators import CalculatorError
    from ezcal.structures import from_ase, to_ase

    try:
        from mc import MaterialMonteCalro
    except ImportError as exc:                       # pragma: no cover - 依存の有無
        raise DynamicsError(
            "material-mc がインストールされていません:  "
            "uv pip install -e ~/repos/material_monte_carlo/mc") from exc

    spec = get_mode(mode or _get(config, "mode", "md"))
    rundir = Path(rundir)
    rundir.mkdir(parents=True, exist_ok=True)

    atoms = to_ase(structure)
    supercell = _get(config, "supercell")
    if supercell:
        atoms = atoms.repeat([int(n) for n in supercell])
        log(f"    スーパーセル {tuple(int(n) for n in supercell)} -> {len(atoms)} 原子")
    atoms = sanitize_cell(atoms)
    atoms, vacancies = make_vacancies(atoms, config, spec, log)

    try:
        calculator = calculators.get_calculator(config)
    except CalculatorError as exc:
        raise DynamicsError(str(exc)) from exc
    description = calculators.describe(config)
    log(f"    calculator: {description}")

    seed = _get(config, "seed")
    driver = MaterialMonteCalro(
        atoms,
        calculator=calculator,
        workdir=rundir,
        temperature_K=float(_get(config, "temperature_K", 300.0)),
        energy_mode=str(_get(config, "energy_mode", "total")),
        n_swap=int(_get(config, "n_swap", 1)),
        swap_min_distance=float(_get(config, "swap_min_distance", 6.0)),
        local_radius=(None if _get(config, "local_radius") is None
                      else float(_get(config, "local_radius"))),
        refresh_interval=int(_get(config, "refresh_interval", 10)),
        swap_pairs=_normalize_species(_get(config, "swap_pairs")),
        species_list=_normalize_species(_get(config, "species")),
        save_interval=int(_get(config, "save_interval", 10)),
        is_shuffle=bool(_get(config, "shuffle", False)),
        seed=None if seed is None else int(seed),
        verbose=bool(_get(config, "verbose", False)),
        progress=bool(_get(config, "progress", True)),
    )

    arguments = build_arguments(config, spec, driver.atoms)
    if spec.kind == "event" and vacancies is not None:
        arguments["vacancy_positions"] = vacancies

    result = DynamicsResult(mode=spec.name, workdir=rundir, calculator=description,
                            natoms=len(driver.atoms),
                            formula=driver.atoms.get_chemical_formula(),
                            parameters=_jsonable(arguments))
    log(f"    {spec.title}: " + ", ".join(f"{k}={v}" for k, v in arguments.items()
                                          if k != "vacancy_positions"))

    start = time.time()
    try:
        result.energy_initial = float(driver.atoms.get_potential_energy())
        outcome = getattr(driver, spec.method)(**arguments)
    except Exception as exc:
        result.messages.append(f"{spec.name} の実行に失敗しました: {exc}")
        result.walltime = time.time() - start
        _write_outputs(result, config, driver, None)
        raise DynamicsError(f"{spec.name} の実行に失敗しました: {exc}") from exc
    result.walltime = time.time() - start

    if isinstance(outcome, dict):                    # event-kMC は dict を返す
        result.event = {k: _jsonable(v) for k, v in outcome.items() if k != "atoms"}
        final_atoms = outcome.get("atoms", driver.atoms)
    else:
        final_atoms = outcome
    result.energy_final = float(driver.atoms.get_potential_energy())
    result.stats = {phase: dict(values) for phase, values in driver.stats.items()}
    for phase, values in result.stats.items():
        if values.get("attempts"):
            values["acceptance"] = values["accepts"] / values["attempts"]

    result.ok = True
    _write_outputs(result, config, driver, from_ase(final_atoms))
    return result


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_outputs(result: DynamicsResult, config, driver, final_structure) -> None:
    """作図・構造・summary.json・report.md をまとめて書き出す。"""
    from ezcal import plotting

    rundir = result.workdir
    records = read_energy_log(rundir / "energy_log.csv")
    result.n_records = len(records)
    result.averages = summarize_records(records)

    if final_structure is not None:
        try:
            final_structure.to(filename=str(rundir / "final_structure.cif"))
            result.exports.append(rundir / "final_structure.cif")
        except Exception as exc:
            result.messages.append(f"final_structure.cif を書けませんでした: {exc}")

    backends = plotting.resolve_backends(config.get("output.plot"))
    if records and backends:
        plots_dir = rundir / "plots"
        try:
            result.plots += plotting.plot_dynamics(
                records, plots_dir, backends,
                dpi=int(config.get("output.dpi", 200)),
                title=f"{result.formula} {result.mode}")
        except Exception as exc:
            result.messages.append(f"作図に失敗しました: {exc}")

    if bool(_get(config, "combine_traj", True)):
        try:
            result.exports.append(Path(driver.combine_traj()))
        except Exception as exc:
            result.messages.append(f"traj を結合できませんでした: {exc}")
    if bool(_get(config, "view_notebook", False)):
        try:
            result.exports.append(Path(driver.create_view_notebook(combine=False)))
        except Exception as exc:
            result.messages.append(f"ビューア notebook を作れませんでした: {exc}")

    (rundir / "summary.json").write_text(
        json.dumps(result.summary(), indent=2, default=str), encoding="utf-8")
    (rundir / "report.md").write_text(render_report(result), encoding="utf-8")


# -------------------------------------------------------------- レポート出力
def render_report(result: DynamicsResult) -> str:
    spec = get_mode(result.mode)

    def fmt(value, digits=4, unit=""):
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.{digits}f}{unit}"
        return f"{value}{unit}"

    lines = [f"# {result.formula} — {spec.title}", "",
             f"- モード: `{result.mode}` ({spec.note})",
             f"- 実行ディレクトリ: `{result.workdir}`",
             f"- calculator: `{result.calculator}`",
             f"- 原子数: {result.natoms}",
             f"- 所要時間: {result.walltime:.1f} s",
             f"- 記録点数: {result.n_records}", ""]

    lines += ["## 設定", "", "| パラメータ | 値 |", "|---|---|"]
    for key, value in result.parameters.items():
        if key == "vacancy_positions":
            value = f"{len(value)} 個"
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
                       ("mean_e_total_eV", "平均全エネルギー (後半)"),
                       ("mean_msd_A2", "平均 MSD (後半)")):
        if result.averages.get(key) is not None:
            lines.append(f"| {label} | {fmt(result.averages[key], 4)} |")
    lines.append("")

    if result.stats:
        lines += ["## MC 受理率", "", "| フェーズ | 受理 / 試行 | 受理率 |", "|---|---|---|"]
        for phase, values in result.stats.items():
            ratio = values.get("acceptance")
            lines.append(f"| {phase} | {values.get('accepts', 0)} / "
                         f"{values.get('attempts', 0)} | {fmt(ratio, 3)} |")
        lines.append("")

    if result.event:
        lines += ["## event-kMC", "", "| 量 | 値 |", "|---|---|",
                  f"| 経過実時間 | {result.event.get('time_s', 0):.3e} s |",
                  f"| ホップ回数 | {result.event.get('n_hops')} |",
                  f"| NEB 実行回数 | {result.event.get('n_neb')} |",
                  f"| MSD | {fmt(result.event.get('msd_A2'), 3, ' Å²')} |",
                  f"| 拡散係数 D | {result.event.get('D_cm2_s', 0):.3e} cm²/s |",
                  f"| イオン伝導度 σ | {result.event.get('sigma_S_cm', 0):.3e} S/cm |", ""]
        barriers = result.event.get("barriers_eV") or {}
        if barriers:
            values = sorted(float(v) for v in barriers.values())
            lines.append(f"障壁 {len(values)} 種: "
                         f"{values[0]:.3f} – {values[-1]:.3f} eV "
                         f"(平均 {sum(values) / len(values):.3f} eV)")
            lines.append("")

    if result.plots or result.exports:
        lines += ["## 出力", ""]
        for path in result.plots:
            lines.append(f"- 図 `{Path(path).name}`")
        for path in result.exports:
            lines.append(f"- データ `{Path(path).name}`")
        lines += ["- 記録 `energy_log.csv` / `energy_profile.png`",
                  "- 構造 `structures/` (CIF スナップショットと MD traj)", ""]
    for message in result.messages:
        lines.append(f"> {message}")
    return "\n".join(lines) + "\n"
