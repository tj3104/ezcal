"""``ezcal`` のコマンドラインインターフェース。"""

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
    help="ezcal - コマンド一つで第一原理計算を実行します"
         "(既定のエンジンは Quantum ESPRESSO)。",
)
config_app = typer.Typer(no_args_is_help=True, help="qe_config.yaml の確認と作成")
pseudo_app = typer.Typer(no_args_is_help=True, help="擬ポテンシャルライブラリの補助コマンド")
vasp_app = typer.Typer(no_args_is_help=True,
                       help="VASP 補助コマンド: 記録済み計算を再生する mock-vasp レジストリ")
bench_app = typer.Typer(no_args_is_help=True,
                        help="内蔵ベンチマーク一式を実行し、実験値および "
                             "Materials Project と比較します")
mlip_app = typer.Typer(no_args_is_help=True,
                       help="機械学習ポテンシャル (MLIP / NNP) の切り替え補助")
app.add_typer(config_app, name="config")
app.add_typer(pseudo_app, name="pseudo")
app.add_typer(vasp_app, name="vasp")
app.add_typer(bench_app, name="bench")
app.add_typer(mlip_app, name="mlip")

console = Console()
TASKS = ["scf", "relax", "vc-relax", "nscf", "bands", "dos", "charge", "auto"]


# ------------------------------------------------------------- 補助関数
def _fail(message: str, code: int = 1) -> None:
    console.print(f"[bold red]error[/]: {message}")
    raise typer.Exit(code)


def _parse_mapping(value: Optional[str], cast=float, allow_list: bool = False) -> dict:
    """``"Fe=0.5,O=0.1"`` -> ``{"Fe": 0.5, "O": 0.1}`` に変換する。

    ``allow_list`` を有効にすると "/" が磁気副格子の値の区切りになる:
    ``"Fe=0.6/-0.6"`` -> ``{"Fe": [0.6, -0.6]}``。
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
    """``"Fe"`` -> ``{"Fe": None}``、``"Fe=1.0,Ni=2.0"`` -> 磁気モーメントの大きさ。"""
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


def _parse_species(value: Optional[str]):
    """``"Fe,Pt"`` -> ``["Fe", "Pt"]``、``"Fe,Pt;Li,O"`` -> グループのリスト。"""
    if not value:
        return None
    groups = [g.strip() for g in str(value).split(";") if g.strip()]
    if len(groups) == 1:
        return [item.strip() for item in groups[0].split(",") if item.strip()]
    return [[item.strip() for item in group.split(",") if item.strip()]
            for group in groups]


def _parse_swap_pairs(value: Optional[str]):
    """``"Fe-Pt,Li-O"`` -> ``[["Fe", "Pt"], ["Li", "O"]]``。"""
    if not value:
        return None
    pairs = []
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.replace("/", "-").split("-") if p.strip()]
        if len(parts) != 2:
            raise typer.BadParameter(f"--swap-pairs は Fe-Pt 形式です: {chunk!r}")
        pairs.append(parts)
    return pairs


def _parse_options(pairs: Optional[list[str]]) -> dict:
    """``["modal=mpa", "batch=4"]`` -> ``{"modal": "mpa", "batch": 4}``。"""
    import yaml

    out: dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise typer.BadParameter(f"--calc-option は key=value 形式です: {item!r}")
        key, _, raw = item.partition("=")
        out[key.strip()] = yaml.safe_load(raw)
    return out


def _apply_mlip_options(overrides: dict, kw: dict) -> None:
    """MLIP / calculator に関する共通オプションを設定の上書きに変換する。"""
    for key, dotted in (("mlip_backend", "mlip.backend"), ("model", "mlip.model"),
                        ("device", "mlip.device"), ("calc_script", "mlip.script"),
                        ("calc_factory", "mlip.factory")):
        if kw.get(key) is not None:
            overrides[dotted] = kw[key]
    options = _parse_options(kw.get("calc_option"))
    if options:
        overrides["mlip.options"] = options


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
    put("bands.scheme", kw.get("band_scheme"))
    put("bands.plotter", kw.get("band_plotter"))
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
    if kw.get("no_charge_map"):
        overrides["charge.map3d"] = False
    put("charge.map_source", kw.get("charge_map_source"))
    if kw.get("iso_level"):
        overrides["charge.isosurface_levels"] = [float(v) for v in kw["iso_level"]]
    if kw.get("no_isosurface"):
        overrides["charge.isosurface"] = False
    _apply_mlip_options(overrides, kw)

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
            console.print(f"[yellow]warning[/]: セルの標準化に失敗しました ({exc})")
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
    table.add_row("タスク", task)
    table.add_row("組成式", f"{info['formula']}  ({info['natoms']} 原子)")
    table.add_row("空間群", f"{info.get('spacegroup')} (#{info.get('spacegroup_number')})")
    table.add_row("エンジン", str(cfg.get("engine")))
    table.add_row("スケジューラ", f"{cfg.get('run.scheduler')} (np={cfg.get('run.nproc')})")
    table.add_row("実行ディレクトリ", str(rundir))
    console.print(table)


def _print_result(result) -> None:
    magnetic = any(result.steps[n].magnetization is not None for n in result.order)
    table = Table(title="計算結果", header_style="bold")
    columns = ["ステップ", "成否", "E (eV)", "E/原子 (eV)", "E_F (eV)", "ギャップ (eV)",
               "max|F|", "P (GPa)"]
    if magnetic:
        columns += ["M (uB)", "|M| (uB)"]
    columns.append("時間 (s)")
    for column in columns:
        table.add_column(column, justify="right")
    for name in result.order:
        step = result.steps[name]

        def fmt(value, digits=4):
            return "-" if value is None else f"{value:.{digits}f}"

        row = [name, "[green]成功[/]" if step.ok else "[red]失敗[/]",
               fmt(step.energy, 6), fmt(step.energy_per_atom, 6),
               fmt(step.fermi_energy), fmt(step.band_gap, 3),
               fmt(step.max_force), fmt(step.pressure, 2)]
        if magnetic:
            row += [fmt(step.magnetization, 3), fmt(step.abs_magnetization, 3)]
        row.append(fmt(step.walltime, 1))
        table.add_row(*row)
    console.print(table)
    for path in result.plots:
        console.print(f"  図      [green]{path}[/]")
    for path in result.exports:
        console.print(f"  データ  [green]{path}[/]")
    console.print(f"  レポート [green]{result.rundir / 'report.md'}[/]")
    console.print(f"  サマリ   [green]{result.rundir / 'summary.json'}[/]")
    for message in result.messages:
        console.print(f"  [yellow]note[/]: {message}")


# --------------------------------------------------------- コマンド本体
def _run_task(task: str, structure_arg: str, kw: dict) -> None:
    from ezcal.workflows import Workflow

    cfg = _build_config(**kw)
    primitive = kw.get("primitive", True)
    if cfg.get("dft.magnetic_sublattices") and primitive:
        # プリミティブセルに縮約すると、磁気秩序の表現に必要なサイトそのものが
        # 失われてしまうため、ここでは縮約を行わない
        primitive = False
        console.print("[yellow]note[/]: 磁気副格子が指定されたため、秩序を表現できるよう "
                      "セルを与えられたまま使用します (--as-is 相当)")
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
    console.print(f"[dim]    実行計画: {' -> '.join(steps) or '(実行するものがありません)'}[/]")

    try:
        result = workflow.run(task, skip=skip, only=bool(kw.get("only")))
    except (StructureError, EngineError, SchedulerError) as exc:
        _fail(str(exc))
    cfg.save(rundir / "qe_config.used.yaml")     # 再現性のため、解決済みの値を保存する

    _print_result(result)
    if not result.ok:
        raise typer.Exit(1)


def _make_command(task: str):
    def command(
        structure: str = typer.Argument(..., metavar="STRUCTURE",
                                        help="CIF/POSCAR/xyz ファイル、または Materials Project ID"),
        # --- DFT ---------------------------------------------------------
        ecutwfc: Optional[float] = typer.Option(None, help="波動関数のカットオフ (Ry)"),
        ecutrho: Optional[float] = typer.Option(None, help="電荷密度のカットオフ (Ry)"),
        kmesh: Optional[tuple[int, int, int]] = typer.Option(
            None, "--kmesh", help="Monkhorst-Pack メッシュ。例: --kmesh 8 8 8"),
        kspacing: Optional[float] = typer.Option(
            None, help="--kmesh 未指定時に使う逆格子空間の間隔 (1/Ang)"),
        functional: Optional[str] = typer.Option(
            None, "--functional", "-f", help="pbe | pbesol | pz (擬ポテンシャルの選択も連動)"),
        input_dft: Optional[str] = typer.Option(None, help="QE の input_dft を直接上書きする"),
        occupations: Optional[str] = typer.Option(
            None, help="smearing | fixed | tetrahedra"),
        smearing: Optional[str] = typer.Option(None, help="mv | gaussian | mp | fd"),
        degauss: Optional[float] = typer.Option(None, help="スメアリング幅 (Ry)"),
        conv_thr: Optional[float] = typer.Option(None, "--conv-thr", help="SCF 収束条件 (Ry)"),
        nbnd: Optional[int] = typer.Option(None, help="バンド数"),
        spin: bool = typer.Option(False, "--spin", "--magnetic",
                                  help="共線スピン分極を有効にする (nspin=2)"),
        magmom: Optional[str] = typer.Option(
            None, help="初期磁化。例: --magmom Fe=0.6,O=0 / "
                       "副格子は '/' で区切る。例: --magmom Fe=0.6/-0.6"),
        afm: Optional[str] = typer.Option(
            None, "--afm",
            help="反強磁性: 指定元素を +/- の副格子に分割する。"
                 "例: --afm Cr または --afm Ni=2.0"),
        hubbard_u: Optional[str] = typer.Option(
            None, "--hubbard-u", help="DFT+U の U 値。例: --hubbard-u Fe=4.0"),
        vdw: Optional[str] = typer.Option(None, help="vdw_corr の指定。例: grimme-d3"),
        # --- 擬ポテンシャル ----------------------------------------------
        pseudo_dir: Optional[str] = typer.Option(None, help="ローカルの UPF ディレクトリ"),
        pseudo: Optional[str] = typer.Option(
            None, help="UPF ファイルを固定指定する。例: --pseudo Fe=Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"),
        # --- バンド / 状態密度 -------------------------------------------
        line_density: Optional[float] = typer.Option(
            None, "--line-density", help="バンド経路上の k 点密度 (1/Ang あたり)"),
        band_scheme: Optional[str] = typer.Option(
            None, "--band-scheme",
            help="高対称 k 経路の決め方: materials_project (既定) | latimer_munro "
                 "| setyawan_curtarolo | seekpath"),
        band_plotter: Optional[str] = typer.Option(
            None, "--band-plotter",
            help="バンド図の描画器: auto (既定) | bsplotter | ezcal"),
        emin: Optional[float] = typer.Option(None, help="プロット窓の下限 (E_F 基準の eV)"),
        emax: Optional[float] = typer.Option(None, help="プロット窓の上限 (E_F 基準の eV)"),
        # --- 実行 --------------------------------------------------------
        nproc: Optional[int] = typer.Option(None, "--np", "-n", help="MPI プロセス数"),
        scheduler: Optional[str] = typer.Option(None, help="local | qsub"),
        qsub: bool = typer.Option(False, "--qsub", help="--scheduler qsub の短縮形"),
        qsub_wait: bool = typer.Option(False, "--qsub-wait",
                                       help="投入したジョブが終わるまで待機する"),
        qsub_script: Optional[str] = typer.Option(None, "--qsub-script",
                                                  help="ジョブスクリプトの雛形 (run_qe.sh)"),
        queue: Optional[str] = typer.Option(None, help="qsub のキュー名"),
        walltime: Optional[str] = typer.Option(None, help="qsub の walltime"),
        nodes: Optional[int] = typer.Option(None, help="qsub のノード数"),
        ppn: Optional[int] = typer.Option(None, help="qsub の 1 ノードあたりプロセス数"),
        dry_run: bool = typer.Option(False, "--dry-run",
                                     help="入力ファイルとジョブスクリプトだけ生成し、計算は行わない"),
        # --- ワークフロー ------------------------------------------------
        only: bool = typer.Option(False, "--only",
                                  help="前段のステップを省き、このステップだけ実行する"),
        skip: Optional[str] = typer.Option(None, help="スキップするステップ (カンマ区切り)"),
        relax: bool = typer.Option(True, "--relax/--no-relax",
                                   help="(auto 用) 先に構造最適化を行う"),
        fixed_cell: bool = typer.Option(False, "--fixed-cell",
                                        help="(auto 用) セルは固定し、原子位置のみ最適化する"),
        with_charge: bool = typer.Option(False, "--charge",
                                         help="(auto 用) 電荷密度の出力と原子電荷の計算も行う"),
        primitive: bool = typer.Option(True, "--primitive/--as-is",
                                       help="事前にプリミティブセルへ標準化する"),
        # --- 出力 --------------------------------------------------------
        outdir: Optional[str] = typer.Option(None, "--outdir", "-o", help="結果の出力先ルート"),
        name: Optional[str] = typer.Option(None, help="実行ディレクトリ名"),
        plot: Optional[str] = typer.Option(
            None, help="matplotlib | plotly | both | none"),
        dpi: Optional[int] = typer.Option(None, help="ラスタ画像の解像度"),
        keep_wfc: bool = typer.Option(False, "--keep-wfc", help="波動関数ファイルを残す"),
        charge_kinds: Optional[str] = typer.Option(
            None, "--charge-kinds",
            help="pp.x で出力する量: density,spin,ae_valence,ae_total,potential"),
        no_bader: bool = typer.Option(False, "--no-bader",
                                      help="電荷密度の Bader 分割を行わない"),
        no_isosurface: bool = typer.Option(False, "--no-isosurface",
                                           help="電荷密度の 3D 等値面を描かない"),
        iso_level: Optional[list[float]] = typer.Option(
            None, "--iso-level",
            help="等値面の値 (e/bohr^3)。複数指定可。既定は分位点から自動"),
        no_charge_map: bool = typer.Option(
            False, "--no-charge-map", help="原子価数の 3D マッピングを描かない"),
        charge_map_source: Optional[str] = typer.Option(
            None, "--charge-map-source",
            help="3D マッピングに使う量: auto | bader_charge | lowdin_charge | "
                 "moment_sphere"),
        # --- MLIP / calculator (engine=mlip のとき) -----------------------
        mlip_backend: Optional[str] = typer.Option(
            None, "--mlip-backend", help="sevennet | mace | chgnet | orb | matgl | emt"),
        model: Optional[str] = typer.Option(
            None, "--model", help="MLIP のモデル名またはチェックポイントのパス"),
        device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda | auto"),
        calc_script: Optional[str] = typer.Option(
            None, "--calc-script", help="calculator を返す自前 Python スクリプト"),
        calc_factory: Optional[str] = typer.Option(
            None, "--calc-factory", help="'module:attribute' 形式の import パス"),
        calc_option: Optional[list[str]] = typer.Option(
            None, "--calc-option", help="calculator への追加引数。例: --calc-option modal=mpa"),
        # --- その他 ------------------------------------------------------
        engine: Optional[str] = typer.Option(None, help="qe | vasp | mlip"),
        config: Optional[str] = typer.Option(None, "--config", "-c",
                                             help="qe_config.yaml のパス"),
        set_options: Optional[list[str]] = typer.Option(
            None, "--set", help="任意の設定キーを上書きする。例: --set dft.mixing_beta=0.2"),
        mp_api_key: Optional[str] = typer.Option(
            None, "--mp-api-key", envvar="MP_API_KEY", help="Materials Project の API キー"),
    ) -> None:
        _run_task(task, structure, dict(locals()))

    command.__name__ = task.replace("-", "_")
    command.__doc__ = {
        "scf": "自己無撞着場 (SCF) 計算を行います。",
        "relax": "原子位置を最適化します。",
        "vc-relax": "原子位置とセルの両方を最適化します。",
        "nscf": "密な k メッシュで非自己無撞着計算を行います (先に scf を実行)。",
        "bands": "自動生成した高対称線に沿ってバンド構造を計算します (先に scf を実行)。",
        "dos": "全状態密度と部分状態密度を計算します (先に scf と nscf を実行)。",
        "charge": "電荷密度の cube ファイルと、Loewdin および Bader の原子電荷を出力します "
                  "(先に scf と dos を実行)。",
        "auto": "構造ファイルだけで一通り実行: vc-relax -> scf -> nscf -> dos -> bands。",
    }[task]
    return command


for _task in TASKS:
    app.command(_task)(_make_command(_task))


# ------------------------------------------------------------ MD / MC
def _build_md_config(**kw) -> Config:
    """``ezcal md`` / ``ezcal mc`` 用の設定を組み立てる。"""
    cfg = load_config(kw.get("config"))
    overrides: dict[str, Any] = {}

    def put(key, value):
        if value is not None:
            overrides[key] = value

    _apply_mlip_options(overrides, kw)
    put("md.mode", kw.get("mode"))
    put("md.temperature_K", kw.get("temperature"))
    put("md.steps", kw.get("steps"))
    put("md.ensemble", kw.get("ensemble"))
    put("md.timestep_fs", kw.get("timestep"))
    put("md.ttime_fs", kw.get("ttime"))
    put("md.pfactor_gpa_fs2", kw.get("pfactor"))
    put("md.pressure_gpa", kw.get("pressure"))
    put("md.tchain", kw.get("tchain"))
    put("md.init_temperature_K", kw.get("init_temperature"))
    put("md.record_interval", kw.get("record_interval"))
    put("md.cycles", kw.get("cycles"))
    put("md.mc_steps", kw.get("mc_steps"))
    put("md.md_steps", kw.get("md_steps"))
    put("md.energy_mode", kw.get("energy_mode"))
    put("md.n_swap", kw.get("n_swap"))
    put("md.save_interval", kw.get("save_interval"))
    put("md.seed", kw.get("seed"))
    put("md.neighbor_cutoff", kw.get("neighbor_cutoff"))
    put("md.r0", kw.get("r0"))
    put("md.voronoi_cutoff", kw.get("voronoi_cutoff"))
    put("md.fmax", kw.get("fmax"))
    put("md.max_itr", kw.get("max_itr"))
    put("md.optimizer", kw.get("optimizer"))
    put("md.mobile_species", kw.get("mobile_species"))
    put("md.vacancy_count", kw.get("vacancies"))
    put("md.nu0_hz", kw.get("nu0"))
    put("md.barrier_method", kw.get("barrier_method"))
    put("md.n_images", kw.get("n_images"))
    put("md.charge", kw.get("ion_charge"))
    put("output.dpi", kw.get("dpi"))

    if kw.get("supercell"):
        overrides["md.supercell"] = [int(v) for v in kw["supercell"]]
    if kw.get("npt_mask"):
        overrides["md.npt_mask"] = [int(v) for v in kw["npt_mask"]]
    if kw.get("vacancy_index"):
        overrides["md.vacancy_indices"] = [int(v) for v in kw["vacancy_index"]]
    species = _parse_species(kw.get("species"))
    if species:
        overrides["md.species"] = species
    pairs = _parse_swap_pairs(kw.get("swap_pairs"))
    if pairs:
        overrides["md.swap_pairs"] = pairs
    if kw.get("shuffle"):
        overrides["md.shuffle"] = True
    if kw.get("reinit_velocities"):
        overrides["md.reinit_velocities"] = True
    if kw.get("no_progress"):
        overrides["md.progress"] = False
    if kw.get("verbose"):
        overrides["md.verbose"] = True
    if kw.get("no_combine_traj"):
        overrides["md.combine_traj"] = False
    if kw.get("view_notebook"):
        overrides["md.view_notebook"] = True
    if kw.get("plot"):
        overrides["output.plot"] = [p.strip() for p in str(kw["plot"]).split(",")]
    for dotted, value in parse_set_options(kw.get("set_options")).items():
        overrides[dotted] = value

    cfg.apply_overrides(overrides)
    return cfg


def _print_dynamics(result) -> None:
    table = Table(title="MD / MC の結果", header_style="bold")
    for column in ("項目", "値"):
        table.add_column(column)

    def fmt(value, digits=4):
        return "-" if value is None else (f"{value:.{digits}f}"
                                          if isinstance(value, float) else str(value))

    table.add_row("モード", result.mode)
    table.add_row("組成 / 原子数", f"{result.formula}  ({result.natoms} 原子)")
    table.add_row("calculator", result.calculator)
    table.add_row("E 初期 (eV)", fmt(result.energy_initial, 6))
    table.add_row("E 最終 (eV)", fmt(result.energy_final, 6))
    for key, label in (("mean_temperature_K", "平均温度 (K)"),
                       ("mean_volume_A3", "平均体積 (A^3)"),
                       ("mean_e_total_eV", "平均全エネルギー (eV)"),
                       ("mean_msd_A2", "平均 MSD (A^2)")):
        if result.averages.get(key) is not None:
            table.add_row(label, fmt(result.averages[key]))
    for phase, values in result.stats.items():
        if values.get("attempts"):
            table.add_row(f"受理率 [{phase}]",
                          f"{values['accepts']}/{values['attempts']} = "
                          f"{values['acceptance']:.3f}")
    if result.event:
        table.add_row("経過実時間 (s)", f"{result.event.get('time_s', 0):.3e}")
        table.add_row("拡散係数 D (cm^2/s)", f"{result.event.get('D_cm2_s', 0):.3e}")
        table.add_row("イオン伝導度 (S/cm)", f"{result.event.get('sigma_S_cm', 0):.3e}")
    table.add_row("所要時間 (s)", fmt(result.walltime, 1))
    console.print(table)
    for path in result.plots:
        console.print(f"  図      [green]{path}[/]")
    for path in result.exports:
        console.print(f"  データ  [green]{path}[/]")
    console.print(f"  レポート [green]{result.workdir / 'report.md'}[/]")
    console.print(f"  サマリ   [green]{result.workdir / 'summary.json'}[/]")
    for message in result.messages:
        console.print(f"  [yellow]note[/]: {message}")


def _make_dynamics_command(default_mode: str, doc: str):
    def command(
        structure: str = typer.Argument(..., metavar="STRUCTURE",
                                        help="CIF/POSCAR/xyz ファイル、または Materials Project ID"),
        mode: str = typer.Option(default_mode, "--mode", "-m",
                                 help="md | mcmc | mcmd | mcrelax | kmc | kmc-voronoi | "
                                      "kmcmd | kmc-voronoi-md | event-kmc"),
        # --- 系の設定 -----------------------------------------------------
        temperature: Optional[float] = typer.Option(
            None, "--temperature", "-T", help="温度 (K)。MD の目標温度 / MC の判定温度"),
        supercell: Optional[tuple[int, int, int]] = typer.Option(
            None, "--supercell", help="入力構造を繰り返す。例: --supercell 2 2 2"),
        seed: Optional[int] = typer.Option(None, "--seed", help="乱数シード"),
        # --- MD -----------------------------------------------------------
        steps: Optional[int] = typer.Option(None, "--steps", "-s",
                                            help="MD / MC のステップ数"),
        ensemble: Optional[str] = typer.Option(
            None, "--ensemble", "-e",
            help="nve | nvt | npt | nph (N P V T E のどれを固定するか)"),
        timestep: Optional[float] = typer.Option(None, "--timestep", help="MD 時間刻み (fs)"),
        ttime: Optional[float] = typer.Option(None, "--ttime", help="熱浴の特性時間 (fs)"),
        pressure: Optional[float] = typer.Option(None, "--pressure", help="目標圧力 (GPa)"),
        pfactor: Optional[float] = typer.Option(None, "--pfactor",
                                                help="圧浴定数 (GPa fs^2)"),
        npt_mask: Optional[tuple[int, int, int]] = typer.Option(
            None, "--npt-mask", help="可変にするセル軸。例: --npt-mask 0 0 1"),
        tchain: Optional[int] = typer.Option(None, "--tchain",
                                             help="Nose-Hoover チェーン長"),
        init_temperature: Optional[float] = typer.Option(
            None, "--init-temperature", help="初期速度の温度 (K)"),
        reinit_velocities: bool = typer.Option(False, "--reinit-velocities",
                                               help="開始時に速度を振り直す"),
        record_interval: Optional[int] = typer.Option(
            None, "--record-interval", help="MD / 緩和中の記録間隔"),
        # --- MC -----------------------------------------------------------
        cycles: Optional[int] = typer.Option(None, "--cycles", "-c",
                                             help="サイクル型モードの繰り返し回数"),
        mc_steps: Optional[int] = typer.Option(None, "--mc-steps",
                                               help="1 サイクルあたりの MC ステップ数"),
        md_steps: Optional[int] = typer.Option(None, "--md-steps",
                                               help="1 サイクルあたりの MD ステップ数"),
        energy_mode: Optional[str] = typer.Option(None, "--energy-mode",
                                                  help="total | peratom"),
        n_swap: Optional[int] = typer.Option(None, "--n-swap",
                                             help="peratom モードでの同時スワップ数"),
        species: Optional[str] = typer.Option(
            None, "--species",
            help="スワップ対象元素。'Fe,Pt' はまとめて 1 群、'Fe,Pt;Li,O' は群ごとに独立"),
        swap_pairs: Optional[str] = typer.Option(
            None, "--swap-pairs", help="交換を許すペア。例: --swap-pairs Fe-Pt,Li-O"),
        shuffle: bool = typer.Option(False, "--shuffle",
                                     help="初期配置の元素をランダムに並べ替える"),
        save_interval: Optional[int] = typer.Option(
            None, "--save-interval", help="構造スナップショットの保存間隔"),
        neighbor_cutoff: Optional[float] = typer.Option(
            None, "--neighbor-cutoff", help="kmc / kmcmd の近接カットオフ (Ang)"),
        r0: Optional[float] = typer.Option(None, "--r0",
                                           help="kmc-voronoi のガウス重み距離 (Ang)"),
        voronoi_cutoff: Optional[float] = typer.Option(
            None, "--voronoi-cutoff", help="ボロノイ分割の点群半径 (Ang)。0 で自動"),
        fmax: Optional[float] = typer.Option(None, "--fmax",
                                             help="mcrelax の力の収束条件 (eV/Ang)"),
        max_itr: Optional[int] = typer.Option(None, "--max-itr", help="緩和の最大反復数"),
        optimizer: Optional[str] = typer.Option(None, "--optimizer",
                                                help="FIRE | BFGS | LBFGS"),
        # --- event-kMC -----------------------------------------------------
        mobile_species: Optional[str] = typer.Option(
            None, "--mobile-species", help="event-kmc の可動元素。例: --mobile-species Li"),
        vacancies: Optional[int] = typer.Option(
            None, "--vacancies", help="可動元素から無作為に作る空孔の数"),
        vacancy_index: Optional[list[int]] = typer.Option(
            None, "--vacancy-index", help="空孔にする原子の番号 (0 始まり、複数指定可)"),
        nu0: Optional[float] = typer.Option(None, "--nu0", help="試行頻度 (Hz)"),
        barrier_method: Optional[str] = typer.Option(None, "--barrier-method",
                                                     help="neb | rrp"),
        n_images: Optional[int] = typer.Option(None, "--n-images",
                                               help="NEB の中間イメージ数"),
        ion_charge: Optional[float] = typer.Option(
            None, "--ion-charge", help="伝導度換算に使う可動イオンの形式電荷"),
        # --- calculator -----------------------------------------------------
        mlip_backend: Optional[str] = typer.Option(
            None, "--mlip-backend", help="sevennet | mace | chgnet | orb | matgl | emt"),
        model: Optional[str] = typer.Option(
            None, "--model", help="モデル名またはチェックポイントのパス。例: 7net-l3i5"),
        device: Optional[str] = typer.Option(None, "--device", help="cpu | cuda | auto"),
        calc_script: Optional[str] = typer.Option(
            None, "--calc-script", help="calculator を返す自前 Python スクリプト"),
        calc_factory: Optional[str] = typer.Option(
            None, "--calc-factory", help="'module:attribute' 形式の import パス"),
        calc_option: Optional[list[str]] = typer.Option(
            None, "--calc-option", help="calculator への追加引数。例: --calc-option modal=mpa"),
        # --- 出力 -----------------------------------------------------------
        outdir: Optional[str] = typer.Option(None, "--outdir", "-o", help="出力先ルート"),
        name: Optional[str] = typer.Option(None, "--name", help="実行ディレクトリ名"),
        plot: Optional[str] = typer.Option(None, "--plot",
                                           help="matplotlib | plotly | both | none"),
        dpi: Optional[int] = typer.Option(None, "--dpi"),
        view_notebook: bool = typer.Option(False, "--view-notebook",
                                           help="nglview で再生する notebook を作る"),
        no_combine_traj: bool = typer.Option(False, "--no-combine-traj",
                                             help="combined.traj を作らない"),
        no_progress: bool = typer.Option(False, "--no-progress", help="進捗バーを出さない"),
        verbose: bool = typer.Option(False, "--verbose",
                                     help="material-mc のログを標準出力にも出す"),
        primitive: bool = typer.Option(False, "--primitive/--as-is",
                                       help="プリミティブセルへ標準化してから始める"),
        config: Optional[str] = typer.Option(None, "--config", "-c",
                                             help="qe_config.yaml のパス"),
        set_options: Optional[list[str]] = typer.Option(
            None, "--set", help="任意の設定キーを上書きする。例: --set md.tchain=5"),
        mp_api_key: Optional[str] = typer.Option(
            None, "--mp-api-key", envvar="MP_API_KEY", help="Materials Project の API キー"),
    ) -> None:
        from ezcal.md import DynamicsError, get_mode, run_dynamics

        kw = dict(locals())
        cfg = _build_md_config(**kw)
        try:
            spec = get_mode(cfg.get("md.mode", default_mode))
        except DynamicsError as exc:
            _fail(str(exc))
        struct = _load_structure(structure, primitive, mp_api_key)
        root = Path(outdir) if outdir else Path(str(cfg.get("output.dir", "ezcal_out")))
        label = name or f"{struct.composition.reduced_formula}_{spec.name}"
        rundir = (root / label).resolve()

        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column()
        table.add_row("モード", f"{spec.name} — {spec.title}")
        table.add_row("組成式", struct.composition.reduced_formula)
        table.add_row("温度", f"{cfg.get('md.temperature_K')} K")
        table.add_row("実行ディレクトリ", str(rundir))
        console.print(table)

        try:
            result = run_dynamics(cfg, struct, rundir, mode=spec.name,
                                  log=lambda msg: console.print(f"[dim]{msg}[/]"))
        except (DynamicsError, StructureError) as exc:
            _fail(str(exc))
        cfg.save(rundir / "qe_config.used.yaml")
        _print_dynamics(result)
        if not result.ok:
            raise typer.Exit(1)

    command.__name__ = f"dynamics_{default_mode}"
    command.__doc__ = doc
    return command


app.command("md")(_make_dynamics_command(
    "md", "分子動力学を実行します (NVE/NVT/NPT/NPH)。material-mc + ASE calculator。"))
app.command("mc")(_make_dynamics_command(
    "mcmc", "モンテカルロを実行します (--mode で MCMC / MCMD / kMC / event-kMC を選択)。"))


# --------------------------------------------------------- mlip グループ
@mlip_app.command("list")
def mlip_list(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """使える MLIP レシピと、この環境で読み込めるかどうかを表示します。"""
    from ezcal import calculators

    cfg = load_config(config)
    extra = cfg.get("mlip.recipe_dirs") or []
    table = Table(header_style="bold")
    for column in ("名前", "別名", "状態", "よく使うモデル", "説明"):
        table.add_column(column, overflow="fold")
    for recipe in calculators.available(extra):
        problems = recipe.check()
        status = "[green]利用可[/]" if not problems else "[yellow]要インストール[/]"
        table.add_row(recipe.name, ", ".join(recipe.aliases) or "-", status,
                      ", ".join(recipe.models[:4]) or "-", recipe.description)
    console.print(table)
    console.print(f"[dim]現在の設定: {calculators.describe(cfg)}[/]")
    console.print("[dim]自前のポテンシャルは --calc-script / --calc-factory、"
                  "または mlip.recipe_dirs にレシピを置いて使えます[/]")


@mlip_app.command("check")
def mlip_check(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    mlip_backend: Optional[str] = typer.Option(None, "--mlip-backend"),
    model: Optional[str] = typer.Option(None, "--model"),
    device: Optional[str] = typer.Option(None, "--device"),
    calc_script: Optional[str] = typer.Option(None, "--calc-script"),
    calc_factory: Optional[str] = typer.Option(None, "--calc-factory"),
    calc_option: Optional[list[str]] = typer.Option(None, "--calc-option"),
    build: bool = typer.Option(False, "--build",
                              help="実際に calculator を作り、H2 分子で 1 点計算する"),
) -> None:
    """いまの設定でどの calculator が使われるかを確認します。"""
    from ezcal import calculators

    cfg = load_config(config)
    overrides: dict[str, Any] = {}
    _apply_mlip_options(overrides, dict(locals()))
    cfg.apply_overrides(overrides)

    console.print(f"設定: [cyan]{calculators.describe(cfg)}[/]")
    problems = calculators.check(cfg)
    for problem in problems:
        console.print(f"[yellow]{problem}[/]")
    if problems:
        raise typer.Exit(1)
    console.print("[green]依存関係は揃っています[/]")
    if not build:
        return
    try:
        calculator = calculators.get_calculator(cfg)
    except Exception as exc:
        _fail(str(exc))
    console.print(f"calculator: [green]{type(calculator).__name__}[/]")
    from ase.build import molecule

    atoms = molecule("H2")
    atoms.center(vacuum=4.0)
    atoms.pbc = True
    atoms.calc = calculator
    try:
        console.print(f"H2 のエネルギー: {float(atoms.get_potential_energy()):.4f} eV")
    except Exception as exc:
        console.print(f"[yellow]1 点計算に失敗しました ({exc})[/] "
                      "- 対応元素の範囲を確認してください")


@mlip_app.command("template")
def mlip_template(
    out: str = typer.Option("my_potential.py", "--out", "-o", help="書き出し先"),
    force: bool = typer.Option(False, "--force", help="既存ファイルを上書きする"),
) -> None:
    """自前ポテンシャル用の calculator スクリプト雛形を書き出します。"""
    source = Path(__file__).with_name("templates") / "calculator_template.py"
    target = Path(out)
    if target.exists() and not force:
        _fail(f"{target} は既に存在します (--force で上書き)")
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"[green]{target}[/] を書き出しました")
    console.print("[dim]build() を書き換えたあと、"
                  f"ezcal md STRUCTURE --calc-script {target} で使えます[/]")


@mlip_app.command("modes")
def mlip_modes() -> None:
    """MD / MC で選べるモードの一覧を表示します。"""
    from ezcal.md import MODE_NAMES, MODES

    table = Table(header_style="bold")
    for column in ("モード", "内容", "説明"):
        table.add_column(column, overflow="fold")
    for name in MODE_NAMES:
        spec = MODES[name]
        table.add_row(spec.name, spec.title, spec.note)
    console.print(table)


# ------------------------------------------------------- config グループ
@config_app.command("init")
def config_init(
    path: str = typer.Option("qe_config.yaml", "--path", "-p", help="書き出し先のパス"),
    force: bool = typer.Option(False, "--force", help="既存ファイルを上書きする"),
    user: bool = typer.Option(False, "--user", help="~/.config/ezcal/ に書き出す"),
) -> None:
    """編集して使える、コメント付きの qe_config.yaml を書き出します。"""
    from ezcal.config import PACKAGE_DEFAULT, USER_CONFIG_PATH

    target = USER_CONFIG_PATH if user else Path(path)
    if target.exists() and not force:
        _fail(f"{target} は既に存在します (上書きするには --force)")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(PACKAGE_DEFAULT.read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"書き出しました: [green]{target}[/]")


@config_app.command("show")
def config_show(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    flat: bool = typer.Option(False, "--flat", help="ドット区切りのキーを 1 行ずつ表示する"),
) -> None:
    """実際に適用される設定と、その読み込み元を表示します。"""
    cfg = load_config(config)
    console.print("[dim]# 読み込み元 (後ろのものが優先):[/]")
    for source in cfg.sources:
        console.print(f"[dim]#   {source}[/]")
    if flat:
        for key, value in flatten(cfg.data):
            console.print(f"{key} = {value}")
    else:
        console.print(cfg.to_yaml())


# ------------------------------------------------------- pseudo グループ
@pseudo_app.command("list")
def pseudo_list(
    element: str = typer.Argument(..., help="元素記号。例: Fe"),
    functional: str = typer.Option("pbe", "--functional", "-f"),
    relativistic: bool = typer.Option(False, "--relativistic"),
) -> None:
    """指定元素について ezcal が把握している擬ポテンシャルを一覧表示します。"""
    from ezcal.pseudo import candidates, rank_candidates

    files = rank_candidates(element, candidates(element, functional, relativistic))
    if not files:
        _fail(f"{element} の {functional} 擬ポテンシャルは登録されていません")
    for i, name in enumerate(files):
        marker = "[green]*[/]" if i == 0 else " "
        console.print(f" {marker} {name}")


@pseudo_app.command("fetch")
def pseudo_fetch(
    elements: list[str] = typer.Argument(..., help="元素記号"),
    functional: str = typer.Option("pbe", "--functional", "-f"),
    pseudo_dir: Optional[str] = typer.Option(None, "--pseudo-dir"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """擬ポテンシャルをローカルライブラリへダウンロードし、推奨カットオフを表示します。"""
    from ezcal.pseudo import PseudoManager

    cfg = load_config(config)
    manager = PseudoManager(
        pseudo_dir=pseudo_dir or cfg.get("qe.pseudo_dir", "~/.ezcal/pseudo"),
        functional=functional,
        preference=cfg.get("qe.pseudo_preference", ["kjpaw", "rrkjus"]),
    )
    table = Table(header_style="bold")
    for column in ("元素", "ファイル", "種別", "Z_val", "ecutwfc (Ry)", "ecutrho (Ry)"):
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
    console.print(f"この組み合わせでの推奨値: ecutwfc={wfc} Ry, ecutrho={rho} Ry")
    console.print(f"ライブラリ: [green]{manager.pseudo_dir}[/]")


# --------------------------------------------------------- bench グループ
@bench_app.command("list")
def bench_list(
    which: str = typer.Option("all", "--set", "-s", help="metals | oxides | all"),
) -> None:
    """ベンチマーク一式に含まれる系と、その参照値を表示します。"""
    from ezcal.benchmark import load_suite

    try:
        entries = load_suite(which)
    except Exception as exc:
        _fail(str(exc))
    table = Table(header_style="bold")
    for column in ("系", "組成式", "構造型", "原子数", "a 実験 (A)", "c 実験 (A)",
                   "ギャップ 実験 (eV)", "B0 実験 (GPa)", "m 実験 (uB)", "スピン"):
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
    console.print(f"[dim]{len(entries)} 系[/]")


@bench_app.command("run")
def bench_run(
    which: str = typer.Option("all", "--set", "-s", help="metals | oxides | all"),
    outdir: str = typer.Option("03_qe_bench", "--outdir", "-o", help="結果の出力先ルート"),
    only: Optional[str] = typer.Option(None, "--only",
                                       help="実行する系の名前 (カンマ区切り)"),
    limit: Optional[int] = typer.Option(None, "--limit", help="N 系実行したら打ち切る"),
    task: Optional[str] = typer.Option(None, "--task", help="タスクを上書きする (既定は auto)"),
    nproc: Optional[int] = typer.Option(None, "--np", "-n", help="MPI プロセス数"),
    engine: str = typer.Option("qe", "--engine", help="qe | vasp | mlip"),
    eos: bool = typer.Option(True, "--eos/--no-eos",
                             help="体積弾性率のため状態方程式のフィッティングも行う"),
    eos_points: int = typer.Option(5, "--eos-points", help="E(V) フィットに使う体積点の数"),
    eos_range: float = typer.Option(0.04, "--eos-range", help="E(V) フィットの最大ひずみ"),
    resume: bool = typer.Option(True, "--resume/--restart",
                                help="既に結果がある系はスキップする"),
    plot: str = typer.Option("matplotlib", "--plot", help="matplotlib | plotly | both | none"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    set_options: Optional[list[str]] = typer.Option(None, "--set-config",
                                                    help="任意の設定キーを上書きする"),
) -> None:
    """ベンチマーク一式を実行します (全系だと数時間かかる長い処理です)。"""
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
            _fail(f"未知の系です: {', '.join(sorted(missing))}")
    if task:
        for entry in entries:
            entry.task = task
    if limit:
        entries = entries[:limit]
    if not entries:
        _fail("実行対象がありません")

    strains = np.linspace(-abs(eos_range), abs(eos_range), max(4, eos_points)).tolist()
    root = Path(outdir).resolve()
    console.print(f"[bold]{len(entries)}[/] 系 -> [green]{root}[/]  "
                  f"(エンジン {engine}, np {cfg.get('run.nproc')}, "
                  f"EOS {'あり' if eos else 'なし'})")

    records = run_suite(entries, cfg, root, eos=eos, resume=resume, strains=strains,
                        log=lambda msg: console.print(f"[dim]{msg}[/]"))

    ok = sum(1 for r in records if r.get("ok"))
    console.print(f"\n[bold]{ok}/{len(records)}[/] 系が完了しました。"
                  f"次のコマンドを実行してください: [green]ezcal bench report {root}[/]")
    if ok < len(records):
        raise typer.Exit(1)


@bench_app.command("report")
def bench_report(
    rundir: str = typer.Argument("03_qe_bench", help="ベンチマーク結果のディレクトリ"),
    mp_api_key: Optional[str] = typer.Option(None, "--mp-api-key", envvar="MP_API_KEY",
                                             help="Materials Project とも比較する"),
    plot: str = typer.Option("both", "--plot", help="matplotlib | plotly | both | none"),
    dpi: int = typer.Option(200, "--dpi"),
) -> None:
    """完了したベンチマーク結果を実験値および Materials Project と比較します。"""
    from ezcal.benchmark import compare, fetch_mp, load_results, summarise, write_report

    root = Path(rundir)
    try:
        records = load_results(root)
    except Exception as exc:
        _fail(str(exc))

    mp_data: dict = {}
    if mp_api_key:
        console.print("[dim]Materials Project のデータを検索しています...[/]")
        try:
            mp_data = fetch_mp(records, mp_api_key,
                               log=lambda msg: console.print(f"[dim]{msg}[/]"))
        except Exception as exc:
            console.print(f"[yellow]Materials Project の検索をスキップしました[/]: {exc}")
    else:
        console.print("[dim]MP API キーが未指定のため、実験値とのみ比較します[/]")

    rows = compare(records, mp_data)
    files = write_report(records, rows, root, backends=[plot], dpi=dpi)

    for against, title in (("exp", "実験値"), ("mp", "Materials Project")):
        summary = summarise(rows, against)
        if not summary:
            continue
        table = Table(title=f"{title} に対する平均誤差", header_style="bold")
        for column in ("物理量", "セット", "n", "ME", "MAE", "MRE %", "MARE %"):
            table.add_column(column, justify="right")
        for entry in summary:
            table.add_row(f"{entry['label']} ({entry['unit'] or '-'})", entry["group"],
                          str(entry["n"]), f"{entry['me']:+.4g}", f"{entry['mae']:.4g}",
                          f"{entry['mre']:+.2f}", f"{entry['mare']:.2f}")
        console.print(table)

    console.print(f"  レポート [green]{files['report']}[/]")
    console.print(f"  データ   [green]{files['csv']}[/]")
    for path in files.get("plots", []):
        console.print(f"  図       [green]{path}[/]")


# ---------------------------------------------------------- vasp グループ
def _vasp_engine(config_path: Optional[str], registry: Optional[str] = None):
    from ezcal.engines import get_engine

    cfg = load_config(config_path)
    if registry:
        cfg.set("vasp.mock.registry", registry)
    return cfg, get_engine("vasp", cfg)


@vasp_app.command("record")
def vasp_record(
    rundir: str = typer.Argument(..., help="完了済みの VASP 実行ディレクトリ"),
    name: str = typer.Option(..., "--name", "-n",
                             help="エントリのラベル。例: si-scf/calc-000"),
    registry: Optional[str] = typer.Option(None, "--registry", "-r",
                                           help="レジストリのディレクトリ (既定: vasp.mock.registry)"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """完了済みの VASP 計算をモックレジストリに登録し、mock-vasp で再生できるようにします。

    VASP が実際に動く環境でこのコマンドを実行し、生成されたレジストリをライセンスの
    ない計算機へコピーすれば、同じ入力を ezcal 経由で再実行できます。
    """
    _, engine = _vasp_engine(config, registry)
    try:
        path = engine.record(Path(rundir), name)
    except Exception as exc:
        _fail(str(exc))
    console.print(f"[green]{rundir}[/] を [green]{path}[/] として登録しました")


@vasp_app.command("registry")
def vasp_registry(
    registry: Optional[str] = typer.Option(None, "--registry", "-r"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """mock-vasp が再生できる計算の一覧を表示します。"""
    from aiida_vasp.utils.mock_code import VaspMockRegistry

    _, engine = _vasp_engine(config, registry)
    base = engine.registry_path()
    if base is None:
        _fail("レジストリが設定されていません (vasp.mock.registry を設定するか --registry を指定)")
    reg = VaspMockRegistry(str(base))
    if not reg.reg_name:
        console.print(f"[yellow]エントリが見つかりません:[/] {base}")
        raise typer.Exit(0)
    table = Table(header_style="bold")
    table.add_column("エントリ")
    table.add_column("ハッシュ")
    table.add_column("パス")
    for name, digest in sorted(reg.reg_name.items()):
        table.add_row(name, digest, str(reg.reg_hash[digest]))
    console.print(table)


@vasp_app.command("hash")
def vasp_hash(
    folder: str = typer.Argument(..., help="INCAR/KPOINTS/POSCAR を含むディレクトリ"),
) -> None:
    """入力ディレクトリの mock-vasp ハッシュを表示します (ミスの原因調査に便利)。"""
    from aiida_vasp.utils.mock_code import VaspMockRegistry

    path = Path(folder)
    if not (path / "INCAR").is_file():
        _fail(f"{path} に INCAR がありません")
    console.print(VaspMockRegistry.compute_hash(path))


# ---------------------------------------------------------- その他コマンド
@app.command("info")
def info(
    structure: str = typer.Argument(..., metavar="STRUCTURE"),
    primitive: bool = typer.Option(True, "--primitive/--as-is"),
    kspacing: float = typer.Option(0.25, help="推奨 k メッシュの算出に使う間隔"),
    mp_api_key: Optional[str] = typer.Option(None, "--mp-api-key", envvar="MP_API_KEY"),
    band_path_only: bool = typer.Option(False, "--band-path", help="k 経路も表示する"),
    band_scheme: str = typer.Option(
        "materials_project", "--band-scheme",
        help="k 経路の決め方: materials_project | latimer_munro "
             "| setyawan_curtarolo | seekpath"),
) -> None:
    """構造を調べます: 対称性、推奨 k メッシュ、擬ポテンシャル、カットオフ。"""
    from ezcal.pseudo import PseudoManager
    from ezcal.structures import auto_kmesh, band_path, structure_info

    struct = _load_structure(structure, primitive, mp_api_key)
    data = structure_info(struct)
    console.print_json(json.dumps(data, indent=2))
    console.print(f"推奨 k メッシュ (kspacing={kspacing}): {auto_kmesh(struct, kspacing)}")
    try:
        manager = PseudoManager()
        infos = manager.resolve_all([str(el) for el in struct.composition.elements])
        wfc, rho = PseudoManager.suggest_cutoffs(infos.values())
        for element, item in infos.items():
            console.print(f"  {element:<3s} {item.filename}  "
                          f"({item.pseudo_type}, Z={item.z_valence:g})")
        console.print(f"推奨カットオフ: ecutwfc={wfc} Ry  ecutrho={rho} Ry")
    except Exception as exc:
        console.print(f"[yellow]擬ポテンシャルを取得できません[/]: {exc}")
    if band_path_only:
        path = band_path(struct, scheme=band_scheme)
        console.print(f"k 経路 ({path.scheme}, {path.nkpt} 点): "
                      + " -> ".join(dict.fromkeys(l for _, l in path.labels if l)))


@app.command("mp")
def mp_get(
    material_id: str = typer.Argument(..., help="例: mp-149"),
    out: Optional[str] = typer.Option(None, "--out", "-o", help="出力する CIF のパス"),
    mp_api_key: Optional[str] = typer.Option(None, "--mp-api-key", envvar="MP_API_KEY"),
) -> None:
    """Materials Project から構造をダウンロードし、CIF として保存します。"""
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
    """利用可能なエンジンと、この環境で実行できるかどうかを表示します。"""
    from ezcal.engines import available, get_engine

    cfg = load_config(config)
    table = Table(header_style="bold")
    for column in ("エンジン", "対応タスク", "状態"):
        table.add_column(column)
    for name in ("qe", "vasp", "mlip"):
        engine = get_engine(name, cfg)
        problems = engine.check()
        status = "[green]利用可[/]" if not problems else f"[yellow]{problems[0]}[/]"
        table.add_row(name, ", ".join(engine.supported), status)
    console.print(table)
    console.print(f"[dim]別名: {', '.join(available())}[/]")


def _read_charge_rows(path: Path) -> list[dict]:
    """``atomic_charges.csv`` を辞書の並びとして読む。"""
    import csv

    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key is None or value in ("", None):
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            if row:
                rows.append(row)
    return rows


def _replot_charge(root: Path, plots_dir: Path, backends, dpi: int, label: str,
                   iso_level, map_source: str) -> list[Path]:
    """実行ディレクトリに残っている cube と原子電荷から図を描き直す。"""
    from ezcal import plotting
    from ezcal.charge import read_cube

    written: list[Path] = []
    for cube_path in sorted(root.rglob("*.cube")):
        kind = cube_path.stem
        try:
            cube = read_cube(cube_path)
        except Exception as exc:
            console.print(f"[yellow]{cube_path.name} を読めません ({exc})[/]")
            continue
        written += plotting.plot_charge_profile(cube, plots_dir, backends, kind=kind,
                                                dpi=dpi, title=f"{label} {kind}")
        written += plotting.plot_charge_slice(cube, plots_dir, backends, kind=kind,
                                              dpi=dpi, title=f"{label} {kind}")
        written += plotting.plot_charge_isosurface(
            cube, plots_dir, kind=kind, backends=backends, dpi=dpi,
            levels=iso_level or None, title=f"{label} {kind}")

    for csv_path in sorted(root.rglob("atomic_charges.csv")):
        rows = _read_charge_rows(csv_path)
        if not rows:
            continue
        lattice = None
        for name in ("final_structure.cif", "input_structure.cif"):
            candidate = root / name
            if candidate.is_file():
                from pymatgen.core import Structure

                lattice = Structure.from_file(candidate).lattice.matrix
                break
        try:
            written += plotting.plot_charge_map_3d(
                rows, plots_dir, backends, lattice=lattice, source=map_source,
                dpi=dpi, title=f"{label} atomic charges")
        except Exception as exc:
            console.print(f"[yellow]原子電荷の 3D 図を描けません ({exc})[/]")
        break
    return written


def _replot_dynamics(root: Path, plots_dir: Path, backends, dpi: int,
                     label: str) -> list[Path]:
    """MD / MC の ``energy_log.csv`` から時系列の図を描き直す。"""
    from ezcal import plotting
    from ezcal.md import read_energy_log

    log_path = root / "energy_log.csv"
    if not log_path.is_file():
        return []
    records = read_energy_log(log_path)
    if not records:
        return []
    return plotting.plot_dynamics(records, plots_dir, backends, dpi=dpi, title=label)


@app.command("plot")
def replot(
    rundir: str = typer.Argument(..., help="過去の ezcal 実行ディレクトリ"),
    plot: str = typer.Option("both", help="matplotlib | plotly | both"),
    emin: float = typer.Option(-10.0, help="エネルギー窓の下限 (E_F 基準の eV)"),
    emax: float = typer.Option(10.0, help="エネルギー窓の上限 (E_F 基準の eV)"),
    dpi: int = typer.Option(200),
    title: Optional[str] = typer.Option(None, help="図のタイトル"),
    charge: bool = typer.Option(True, "--charge/--no-charge",
                                help="cube ファイルと原子電荷の図も描き直す"),
    iso_level: Optional[list[float]] = typer.Option(
        None, "--iso-level", help="等値面の値 (e/bohr^3)。複数指定可"),
    charge_map_source: str = typer.Option(
        "auto", "--charge-map-source",
        help="3D マッピングに使う量: auto | bader_charge | lowdin_charge | moment_sphere"),
    band_plotter: str = typer.Option(
        "auto", "--band-plotter",
        help="バンド図の描画器: auto (既定) | bsplotter | ezcal"),
) -> None:
    """完了済みの計算結果から図を描き直します (再計算はしません)。

    バンド / DOS は ``raw_data.json`` から、電荷密度は各ステップの ``*.cube`` と
    ``atomic_charges.csv`` から、MD / MC は ``energy_log.csv`` から描き直します。
    """
    import numpy as np

    from ezcal import plotting
    from ezcal.engines.base import CalcResult

    root = Path(rundir)
    backends = plotting.resolve_backends([p.strip() for p in plot.split(",")])
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    label = title or root.name

    written: list[Path] = []
    if charge:
        written += _replot_charge(root, plots_dir, backends, dpi, label,
                                  iso_level, charge_map_source)
    written += _replot_dynamics(root, plots_dir, backends, dpi, label)

    raw_path = root / "raw_data.json"
    if not raw_path.is_file():
        if written:
            for path in written:
                console.print(f"[green]{path}[/]")
            return
        _fail(f"{raw_path} が見つかりません - 計算をやり直すか、実行ディレクトリを指定してください")
    payload = json.loads(raw_path.read_text())
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

    if bands_result is not None:
        written += plotting.plot_bands(bands_result, plots_dir, backends, fermi=fermi,
                                       emin=emin, emax=emax, dpi=dpi, zero=zero,
                                       title=f"{label} band structure",
                                       plotter=band_plotter)
    if dos_result is not None:
        written += plotting.plot_dos(dos_result, plots_dir, backends, fermi=fermi,
                                     emin=emin, emax=emax, dpi=dpi, zero=zero,
                                     title=f"{label} density of states")
    if bands_result is not None and dos_result is not None:
        written += plotting.plot_bands_dos(bands_result, dos_result, plots_dir, backends,
                                           fermi=fermi, emin=emin, emax=emax, dpi=dpi,
                                           zero=zero, title=label)
    if not written:
        _fail("描画できるものがありません: この計算にはバンドや DOS のデータがありません")
    for path in written:
        console.print(f"[green]{path}[/]")


@app.command("version")
def version() -> None:
    """ezcal のバージョンを表示します。"""
    console.print(f"ezcal {__version__}  (python {sys.version.split()[0]})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
