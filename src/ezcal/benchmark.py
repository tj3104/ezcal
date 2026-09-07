"""ベンチマーク一式: 決められた小さなセル群を計算し、参照値と比較する。

目的は「この ezcal 環境は正しい数値を出しているか」を、新しい計算機上で
コマンド一つで確かめられるようにすること。構造はプロトタイプと実験格子定数から
構築するため、ネットワークも API キーも不要。Materials Project の値は、キーが
利用できる場合にレポート作成時に参照する。

比較する物理量
--------------
========================  ============================  =====================
物理量                    取得元                        比較対象
========================  ============================  =====================
格子定数 a, c, c/a        vc-relax                      実験値, MP
原子あたり体積            vc-relax                      実験値, MP
密度                      vc-relax                      実験値, MP
体積弾性率 B0, B0'        E(V) の Birch-Murnaghan 近似  実験値
バンドギャップ (直接か)   密メッシュでの nscf           実験値, MP
磁気モーメント            scf                           実験値, MP
N(E_F)                    dos.x                         文献値 (金属)
d バンド中心              projwfc.x                     文献値 (金属)
========================  ============================  =====================

凝集エネルギーと生成エネルギーは意図的に除外している。意味のある値にするには
孤立原子や O2 の参照計算と陰イオン補正が必要であり、環境が正常かを確かめるという
本来の目的とは別の作業になるため。
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

SUITE_PATH = Path(__file__).with_name("data") / "benchmarks.yaml"
GROUPS = ("metals", "oxides")


class BenchmarkError(RuntimeError):
    pass


# ------------------------------------------------------------ プロトタイプ
def _hex_lattice(a: float, c: float):
    from pymatgen.core import Lattice

    return Lattice.hexagonal(a, c)


#: 構築したままのセルを保たなければならないプロトタイプ (プリミティブ縮約を
#: すると、磁気秩序に必要なサイトが失われてしまう)
KEEP_AS_BUILT = {"rocksalt_afm2"}


def build_structure(entry: "BenchEntry", primitive: bool = True):
    """プロトタイプと格子定数から初期構造を構築する。

    ``from_spacegroup`` が返すのは従来格子。ベンチマークではプリミティブセルを
    計算するが、反強磁性プロトタイプだけは例外で、大きいセルそのものが磁気単位胞に
    あたるためそのまま使う。
    """
    structure = _build_conventional(entry)
    if not primitive or entry.prototype in KEEP_AS_BUILT:
        return structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return SpacegroupAnalyzer(structure, symprec=1e-4).get_primitive_standard_structure()
    except Exception:
        return structure


def _build_conventional(entry: "BenchEntry"):
    from pymatgen.core import Lattice, Structure

    kind = entry.prototype
    a, c, u = entry.a, entry.c, entry.u
    species = _species(entry.formula)

    if kind in {"fcc", "bcc", "diamond"}:
        group = {"fcc": "Fm-3m", "bcc": "Im-3m", "diamond": "Fd-3m"}[kind]
        return Structure.from_spacegroup(group, Lattice.cubic(a), [species[0]], [[0, 0, 0]])

    if kind == "hcp":
        return Structure.from_spacegroup("P6_3/mmc", _hex_lattice(a, c),
                                         [species[0]], [[1 / 3, 2 / 3, 0.25]])

    if kind == "rocksalt":                                   # AB
        return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), species,
                                         [[0, 0, 0], [0.5, 0.5, 0.5]])

    if kind == "fluorite":                                   # AB2
        return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), species,
                                         [[0, 0, 0], [0.25, 0.25, 0.25]])

    if kind == "antifluorite":                               # A2B
        return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), species,
                                         [[0.25, 0.25, 0.25], [0, 0, 0]])

    if kind == "wurtzite":
        return Structure.from_spacegroup("P6_3mc", _hex_lattice(a, c), species,
                                         [[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, u or 0.375]])

    if kind == "rutile":
        return Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(a, c), species,
                                         [[0, 0, 0], [u or 0.305, u or 0.305, 0]])

    if kind == "anatase":
        return Structure.from_spacegroup("I4_1/amd", Lattice.tetragonal(a, c), species,
                                         [[0, 0, 0], [0, 0, u or 0.208]])

    if kind == "perovskite":                                 # ABO3
        return Structure.from_spacegroup("Pm-3m", Lattice.cubic(a), species,
                                         [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]])

    if kind == "cuprite":                                    # A2B
        return Structure.from_spacegroup("Pn-3m", Lattice.cubic(a), species,
                                         [[0.25, 0.25, 0.25], [0, 0, 0]])

    if kind == "rocksalt_afm2":
        # AFM-II 岩塩型: [111] 方向に 2 倍にした菱面体セル。陽イオンの (111) 面が
        # スピンの向きを交互に取る。化学式単位は 2 つ。
        matrix = a * np.array([[0.5, 0.5, 1.0], [0.5, 1.0, 0.5], [1.0, 0.5, 0.5]])
        cation, anion = species
        return Structure(Lattice(matrix), [cation, cation, anion, anion],
                         [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]])

    raise BenchmarkError(f"未知のプロトタイプです: {kind!r}")


def _species(formula: str) -> list[str]:
    """``"SrTiO3"`` -> ``["Sr", "Ti", "O"]`` (重複を除いた元素を出現順に)。"""
    from pymatgen.core import Composition

    composition = Composition(formula)
    seen: list[str] = []
    import re

    for symbol in re.findall(r"[A-Z][a-z]?", formula):
        if symbol in composition.as_dict() and symbol not in seen:
            seen.append(symbol)
    return seen or [str(el) for el in composition.elements]


# ------------------------------------------------------------ ベンチマーク一式
@dataclass
class BenchEntry:
    name: str
    formula: str
    prototype: str
    group: str
    a: float
    c: float | None = None
    u: float | None = None
    task: str = "auto"
    settings: dict[str, Any] = field(default_factory=dict)
    nspin: int = 1
    magmom: dict[str, float] = field(default_factory=dict)
    afm: dict[str, float] = field(default_factory=dict)
    reference: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def natoms(self) -> int:
        """実際に計算するセルに含まれる原子数。"""
        try:
            return len(build_structure(self))
        except Exception:
            return 0


def load_suite(which: str = "all", path: str | Path | None = None) -> list[BenchEntry]:
    """同梱のベンチマーク定義を読み込む。"""
    data = yaml.safe_load(Path(path or SUITE_PATH).read_text(encoding="utf-8"))
    common = dict(data.get("defaults", {}) or {})
    groups = GROUPS if which in {"all", ""} else tuple(
        g.strip() for g in str(which).split(",") if g.strip())

    entries: list[BenchEntry] = []
    for group in groups:
        if group not in data:
            raise BenchmarkError(f"unknown benchmark group {group!r}; "
                                 f"available: {', '.join(GROUPS)}")
        block = data[group]
        group_defaults = dict(block.get("defaults", {}) or {})
        for raw in block["entries"]:
            raw = dict(raw)
            settings = {**common, **group_defaults}
            for key in ("task", "conv_thr", "kspacing", "line_density", "degauss",
                        "smearing", "occupations", "ecutwfc", "ecutrho"):
                if key in raw:
                    settings[key] = raw.pop(key)
            entries.append(BenchEntry(
                name=raw["name"], formula=raw["formula"], prototype=raw["prototype"],
                group=group, a=float(raw["a"]),
                c=float(raw["c"]) if raw.get("c") is not None else None,
                u=float(raw["u"]) if raw.get("u") is not None else None,
                task=str(settings.pop("task", "auto")),
                settings=settings,
                nspin=int(raw.get("nspin", 1) or 1),
                magmom=dict(raw.get("magmom", {}) or {}),
                afm=dict(raw.get("afm", {}) or {}),
                reference=dict(raw.get("reference", {}) or {}),
                note=str(raw.get("note", "")),
            ))
    return entries


def entry_config(entry: BenchEntry, base):
    """1 エントリ分の設定。ユーザー設定の上にベンチマーク側の設定を重ねる。"""
    cfg = base.update({})
    mapping = {"conv_thr": "dft.conv_thr", "kspacing": "dft.kspacing",
               "degauss": "dft.degauss", "smearing": "dft.smearing",
               "occupations": "dft.occupations", "ecutwfc": "dft.ecutwfc",
               "ecutrho": "dft.ecutrho", "line_density": "bands.line_density"}
    for key, dotted in mapping.items():
        if entry.settings.get(key) is not None:
            cfg.set(dotted, entry.settings[key])
    if entry.nspin == 2 or entry.afm:
        cfg.set("dft.nspin", 2)
    if entry.magmom:
        cfg.set("dft.starting_magnetization", dict(entry.magmom))
    if entry.afm:
        cfg.set("dft.magnetic_sublattices",
                {element: [value, -value] for element, value in entry.afm.items()})
    return cfg


# ------------------------------------------------------ 状態方程式 (B0)
def eos_points(strains: Sequence[float], structure) -> list:
    """``structure`` を等方的にスケールした複製を作る。"""
    out = []
    for strain in strains:
        scaled = structure.copy()
        scaled.scale_lattice(structure.volume * (1.0 + strain))
        out.append(scaled)
    return out


def fit_eos(volumes: Sequence[float], energies: Sequence[float]) -> dict:
    """Birch-Murnaghan フィット -> V0 (A^3)、E0 (eV)、B0 (GPa)、B0'。"""
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if len(volumes) < 4 or not np.all(np.isfinite(energies)):
        return {}
    try:
        from ase.eos import EquationOfState

        eos = EquationOfState(volumes.tolist(), energies.tolist(), eos="birchmurnaghan")
        v0, e0, b0 = eos.fit()
        # ASE が返す B0 の単位は eV/A^3
        result = {"v0": float(v0), "e0": float(e0), "b0": float(b0) * 160.21766208}
        coefficients = getattr(eos, "eos_parameters", None)
        if coefficients is not None and len(coefficients) > 3:
            result["b0_prime"] = float(coefficients[3])
        return result
    except Exception:
        return {}


# ---------------------------------------------------------- DOS 由来の指標
def dos_metrics(dos_payload: Mapping[str, Any] | None,
                pdos_payload: Mapping[str, Any] | None,
                fermi: float | None, natoms: int) -> dict:
    """N(E_F) と d バンド中心。いずれも金属を特徴づける標準的な指標。"""
    out: dict[str, Any] = {}
    if not dos_payload or fermi is None:
        return out
    energy = np.asarray(dos_payload.get("energy", []), dtype=float)
    total = np.asarray(dos_payload.get("dos", []), dtype=float)
    if energy.size < 3 or total.size != energy.size:
        return out

    n_ef = float(np.interp(fermi, energy, total))
    out["n_ef"] = n_ef
    out["n_ef_per_atom"] = n_ef / max(1, natoms)

    if not pdos_payload:
        return out
    channels = pdos_payload.get("per_orbital") or {}
    d_energy = np.asarray(pdos_payload.get("energy", energy), dtype=float)
    d_total = None
    for key, values in channels.items():
        if not key.endswith("-d"):
            continue
        values = np.asarray(values, dtype=float)
        if values.shape != d_energy.shape:
            continue
        d_total = values if d_total is None else d_total + values
    if d_total is None:
        return out
    # 占有部分のみで積分する。一般に引用されるのはこの定義の値
    window = d_energy <= fermi
    weight = d_total[window]
    if weight.sum() <= 0:
        return out
    out["d_band_centre"] = float(np.trapezoid(weight * d_energy[window], d_energy[window])
                                 / np.trapezoid(weight, d_energy[window])) - fermi
    return out


# ------------------------------------------------------------ 構造に関する値
#: 各立方晶プロトタイプの従来立方セルに含まれる原子数。格子定数は原子あたり体積
#: から求める。セルがわずかに歪んでいる場合でも破綻しないのはこの方法だけである
#: (反強磁性体は菱面体的に緩和し、その従来格子は六方晶になるため、"a" は立方セルの
#: 稜ではなく最近接原子間距離になってしまう)。また、手元のセルがプリミティブか
#: 従来格子かにも依存しない。
CUBIC_CONVENTIONAL_ATOMS = {"fcc": 4, "bcc": 2, "diamond": 8, "rocksalt": 8,
                            "fluorite": 12, "antifluorite": 12, "perovskite": 5,
                            "cuprite": 6, "rocksalt_afm2": 8}

#: 相が本当に一致している場合にのみ a と c を比較できるプロトタイプ
NON_CUBIC = {"hcp", "wurtzite", "rutile", "anatase"}


def structure_metrics(structure, prototype: str | None = None) -> dict:
    """緩和後セルの格子定数、原子あたり体積、密度。"""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    lattice = structure.lattice
    out = {
        "natoms": len(structure),
        "volume": float(structure.volume),
        "volume_per_atom": float(structure.volume) / max(1, len(structure)),
        "density": float(structure.density),
    }
    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=1e-3)
        conventional = analyzer.get_conventional_standard_structure()
        out["a"] = float(conventional.lattice.a)
        out["b"] = float(conventional.lattice.b)
        out["c"] = float(conventional.lattice.c)
        out["spacegroup"] = analyzer.get_space_group_symbol()
    except Exception:
        out["a"], out["b"], out["c"] = lattice.a, lattice.b, lattice.c

    conventional_atoms = CUBIC_CONVENTIONAL_ATOMS.get(prototype or "")
    if conventional_atoms:
        edge = (out["volume_per_atom"] * conventional_atoms) ** (1.0 / 3.0)
        out["a"] = out["b"] = out["c"] = float(edge)
        out["a_from_volume"] = True
    elif out.get("a"):
        out["c_over_a"] = out["c"] / out["a"]
    return out


# ------------------------------------------------------------------ 実行部
DEFAULT_STRAINS = (-0.04, -0.02, 0.0, 0.02, 0.04)


def run_eos(entry: BenchEntry, config, structure, workdir: Path,
            strains: Sequence[float] = DEFAULT_STRAINS, log=print) -> dict:
    """等方的にスケールした数点の体積で scf を行い、Birch-Murnaghan でフィットする。"""
    from ezcal.engines import get_engine
    from ezcal.scheduler import get_scheduler

    scheduler = get_scheduler(config)
    engine = get_engine(config.get("engine", "qe"), config, scheduler)
    volumes: list[float] = []
    energies: list[float] = []

    # 全体積で同じ k メッシュを使う。kspacing に点ごとのメッシュを選ばせると E(V) に
    # 段差が生じ、フィットして得られる体積弾性率が意味を成さなくなる
    kmesh = engine._kmesh(config, structure)
    log(f"    EOS: {kmesh[0]}x{kmesh[1]}x{kmesh[2]} メッシュ固定で {len(strains)} 体積点")

    for index, (strain, scaled) in enumerate(zip(strains, eos_points(strains, structure))):
        point_dir = workdir / f"eos_{index:02d}"
        try:
            result = engine.run(scaled, "scf", point_dir, outdir=point_dir / "tmp",
                                kmesh=kmesh)
        except Exception as exc:
            log(f"    eos {strain:+.0%}: {exc}")
            continue
        if not result.ok or result.energy is None:
            log(f"    eos {strain:+.0%}: scf が完了しませんでした")
            continue
        volumes.append(float(scaled.volume))
        energies.append(float(result.energy))
        # EOS の各点から必要なのはエネルギーだけ。1 系あたり 5 回分の追加 scf の
        # 波動関数と電荷密度を残すと、合計で数 GB になってしまう
        if not config.get("output.keep_wavefunctions", False):
            shutil.rmtree(point_dir / "tmp", ignore_errors=True)

    fit = fit_eos(volumes, energies)
    payload = {"strains": list(strains), "volumes": volumes, "energies": energies,
               "kmesh": list(kmesh), **fit}
    if fit:
        payload["v0_per_atom"] = fit["v0"] / max(1, len(structure))
        log(f"    EOS: V0 = {fit['v0']:.3f} A^3, B0 = {fit['b0']:.1f} GPa "
            f"({len(volumes)} 点)")
    return payload


def run_entry(entry: BenchEntry, base_config, root: Path, eos: bool = True,
              strains: Sequence[float] = DEFAULT_STRAINS, log=print) -> dict:
    """ベンチマーク 1 エントリを実行し、比較対象の物理量をすべて集める。"""
    from ezcal.workflows import Workflow

    started = time.time()
    rundir = Path(root) / entry.group / entry.name
    config = entry_config(entry, base_config)
    structure = build_structure(entry)

    record: dict[str, Any] = {
        "name": entry.name, "formula": entry.formula, "group": entry.group,
        "prototype": entry.prototype, "note": entry.note,
        "reference": entry.reference, "rundir": str(rundir),
        "initial": structure_metrics(structure, entry.prototype),
        "settings": {k: v for k, v in entry.settings.items() if v is not None},
        "nspin": 2 if (entry.nspin == 2 or entry.afm) else 1,
    }

    try:
        workflow = Workflow(config, structure, rundir, label=entry.name, log=log)
        result = workflow.run(entry.task)
    except Exception as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                      elapsed_s=round(time.time() - started, 1))
        rundir.mkdir(parents=True, exist_ok=True)
        (rundir / "bench_entry.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8")
        return record

    record["ok"] = bool(result.ok)
    record["steps"] = {name: result.steps[name].summary() for name in result.order}
    record["messages"] = result.messages
    record["ecutwfc"] = config.get("dft.ecutwfc")
    record["ecutrho"] = config.get("dft.ecutrho")
    record["kmesh"] = next((result.steps[n].kmesh for n in result.order
                            if result.steps[n].kmesh), None)

    if result.structure_final is not None:
        record["relaxed"] = structure_metrics(result.structure_final, entry.prototype)

    scf = result.steps.get("scf") or result.steps.get(result.order[0] if result.order else "")
    dense = result.steps.get("bands") or result.steps.get("nscf") or scf
    if scf is not None:
        record["energy"] = scf.energy
        record["energy_per_atom"] = scf.energy_per_atom
        record["magnetization"] = scf.magnetization
        record["abs_magnetization"] = scf.abs_magnetization
        record["site_magnetization"] = scf.site_magnetization
        record["pressure"] = scf.pressure
    if dense is not None:
        record["fermi_energy"] = dense.fermi_energy
        record["band_gap"] = dense.band_gap
        gap_info = dense.data.get("gap_info") or {}
        record["metal"] = bool(gap_info.get("metal")) if gap_info else None
        record["gap_direct"] = gap_info.get("direct")

    dos_step = result.steps.get("dos")
    if dos_step is not None and dos_step.ok:
        reference = (record.get("band_gap") or 0) and record.get("fermi_energy")
        record["dos_metrics"] = dos_metrics(dos_step.data.get("dos"),
                                            dos_step.data.get("pdos"),
                                            record.get("fermi_energy"),
                                            record.get("relaxed", {}).get("natoms", 1))

    if eos and result.ok and result.structure_final is not None:
        record["eos"] = run_eos(entry, config, result.structure_final,
                                rundir / "eos", strains, log)

    record["elapsed_s"] = round(time.time() - started, 1)
    (rundir / "bench_entry.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def run_suite(entries: Iterable[BenchEntry], base_config, root: str | Path,
              eos: bool = True, resume: bool = True,
              strains: Sequence[float] = DEFAULT_STRAINS, log=print) -> list[dict]:
    """全エントリを実行する。終わった順に結果を書き出すため、中断後の再開ができる。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    entries = list(entries)
    records: list[dict] = []

    for index, entry in enumerate(entries, start=1):
        cached = root / entry.group / entry.name / "bench_entry.json"
        if resume and cached.is_file():
            previous = json.loads(cached.read_text(encoding="utf-8"))
            if previous.get("ok"):
                log(f"[{index}/{len(entries)}] {entry.name}: 実行済みのためスキップします")
                records.append(previous)
                continue
            log(f"[{index}/{len(entries)}] {entry.name}: 前回失敗したため再実行します")
        log(f"[{index}/{len(entries)}] {entry.name} ({entry.formula}, {entry.prototype})")
        record = run_entry(entry, base_config, root, eos=eos, strains=strains, log=log)
        status = "成功" if record.get("ok") else f"失敗 ({record.get('error', 'ログを参照')})"
        log(f"    {status}, {record.get('elapsed_s', 0):.0f} 秒")
        records.append(record)
        _write_results(root, records)
    return records


def _write_results(root: Path, records: Sequence[dict]) -> Path:
    path = Path(root) / "bench_results.json"
    path.write_text(json.dumps(list(records), indent=2, default=str), encoding="utf-8")
    return path


def load_results(root: str | Path, refresh: bool = True) -> list[dict]:
    """エントリごとの結果を読み込み、構造由来の指標を再計算する。

    重いのは Quantum ESPRESSO の部分だけなので、対称性解析は読み込み時にやり直す。
    こうしておけば、そこを修正してもベンチマーク全体を再実行せずに済む。
    """
    root = Path(root)
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(root.glob("*/*/bench_entry.json"))]
    if not records:
        combined = root / "bench_results.json"
        if combined.is_file():
            records = json.loads(combined.read_text(encoding="utf-8"))
    if not records:
        raise BenchmarkError(f"{root} にベンチマーク結果がありません")
    if refresh:
        for record in records:
            _refresh_structure_metrics(record)
    return records


def _refresh_structure_metrics(record: dict) -> None:
    from pymatgen.core import Structure

    rundir = Path(record.get("rundir", ""))
    for key, filename in (("initial", "input_structure.json"),
                          ("relaxed", "final_structure.json")):
        path = rundir / filename
        if not path.is_file():
            continue
        try:
            structure = Structure.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        record[key] = structure_metrics(structure, record.get("prototype"))


# --------------------------------------------- Materials Project の照会
MP_FIELDS = ("material_id", "formula_pretty", "symmetry", "structure", "band_gap",
             "is_gap_direct", "volume", "density", "nsites", "total_magnetization",
             "energy_above_hull", "formation_energy_per_atom", "theoretical")


def fetch_mp(records: Sequence[dict], api_key: str, log=print) -> dict[str, dict]:
    """組成式で Materials Project を検索し、空間群で絞り込む。

    material id を直接書き込むのではなく組成式 + 対称性で照合することで、MP 側の
    再インデックスがあってもベンチマークが破綻しないようにしている。
    """
    try:
        from mp_api.client import MPRester
    except ImportError as exc:                                  # pragma: no cover
        raise BenchmarkError("mp-api がインストールされていません: uv pip install mp-api") from exc

    out: dict[str, dict] = {}
    with MPRester(api_key) as mpr:
        for record in records:
            formula = record["formula"]
            wanted = (record.get("initial") or {}).get("spacegroup")
            try:
                docs = mpr.materials.summary.search(formula=formula, fields=list(MP_FIELDS))
            except Exception as exc:
                log(f"    {formula} の MP 検索に失敗しました: {exc}")
                continue
            if not docs:
                continue

            def score(doc):
                symbol = getattr(getattr(doc, "symmetry", None), "symbol", None)
                return (0 if (wanted and symbol == wanted) else 1,
                        float(getattr(doc, "energy_above_hull", None) or 0.0))

            doc = sorted(docs, key=score)[0]
            structure = getattr(doc, "structure", None)
            entry: dict[str, Any] = {
                "material_id": str(getattr(doc, "material_id", "")),
                "spacegroup": getattr(getattr(doc, "symmetry", None), "symbol", None),
                "band_gap": _as_float(getattr(doc, "band_gap", None)),
                "is_gap_direct": getattr(doc, "is_gap_direct", None),
                "density": _as_float(getattr(doc, "density", None)),
                "total_magnetization": _as_float(getattr(doc, "total_magnetization", None)),
                "energy_above_hull": _as_float(getattr(doc, "energy_above_hull", None)),
                "formation_energy_per_atom": _as_float(
                    getattr(doc, "formation_energy_per_atom", None)),
                "spacegroup_wanted": wanted,
            }
            if structure is not None:
                entry.update(structure_metrics(structure, record.get("prototype")))
            out[record["name"]] = entry
            matched = entry["spacegroup"] == entry.get("spacegroup_wanted")
            log(f"    MP {formula}: {entry['material_id']} "
                f"({entry['spacegroup']}" + ("" if matched else " - 別の相") + ")")
    return out


def _as_float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ 比較処理
#: 比較する量、計算値の格納場所、参照値の対応
COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("a", "格子定数 a", "A", "relaxed.a"),
    ("c", "格子定数 c", "A", "relaxed.c"),
    ("c_over_a", "c/a", "", "relaxed.c_over_a"),
    ("volume_per_atom", "原子あたり体積", "A^3", "relaxed.volume_per_atom"),
    ("density", "密度", "g/cm^3", "relaxed.density"),
    ("b0", "体積弾性率", "GPa", "eos.b0"),
    ("gap", "バンドギャップ", "eV", "band_gap"),
    ("magmom", "磁気モーメント", "uB/atom", "site_moment"),
)


def _dig(record: Mapping[str, Any], dotted: str):
    node: Any = record
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _calc_value(record: Mapping[str, Any], key: str, path: str):
    if key == "magmom":
        moments = record.get("site_magnetization") or []
        moments = [abs(float(m)) for m in moments if abs(float(m)) > 0.05]
        return max(moments) if moments else None
    return _as_float(_dig(record, path))


def compare(records: Sequence[dict], mp_data: Mapping[str, dict] | None = None) -> list[dict]:
    """(系, 物理量) ごとに 1 行を作り、計算値・実験値・MP 値を並べる。"""
    rows: list[dict] = []
    for record in records:
        reference = record.get("reference") or {}
        mp_entry = (mp_data or {}).get(record["name"]) or {}
        # MP が採用した相が、こちらの指定したプロトタイプと一致するとは限らない
        # (BaTiO3 は正方晶に、ルチルは Imma に緩和する、など)
        same_phase = bool(mp_entry) and (
            mp_entry.get("spacegroup") == (record.get("initial") or {}).get("spacegroup"))
        for key, label, unit, path in COMPARISONS:
            # c と c/a は非立方晶プロトタイプでのみ意味を持つ
            if key in {"c", "c_over_a"} and reference.get("c") is None:
                continue
            calc = _calc_value(record, key, path)
            exp = _as_float(reference.get(key))
            mp_value = _as_float(_mp_key(mp_entry, key))
            if calc is None and exp is None and mp_value is None:
                continue
            row = {
                "name": record["name"], "group": record["group"],
                "formula": record["formula"], "quantity": key, "label": label,
                "unit": unit, "calc": calc, "exp": exp, "mp": mp_value,
                "mp_id": mp_entry.get("material_id"),
                "mp_spacegroup": mp_entry.get("spacegroup"),
                "mp_same_phase": same_phase if mp_entry else None,
            }
            # 相が異なる場合、その a・c・c/a は同じ物理量とは言えない
            if key in {"a", "c", "c_over_a"} and mp_value is not None and not same_phase:
                row["mp"] = None
                row["mp_excluded"] = "MP は別の相に緩和している"
                mp_value = None
            for other in ("exp", "mp"):
                value = row[other]
                if calc is not None and value not in (None, 0.0):
                    row[f"err_{other}"] = calc - value
                    row[f"relerr_{other}"] = 100.0 * (calc - value) / abs(value)
            rows.append(row)
    return rows


#: 物理量 -> Materials Project レコード上の項目 (None は MP に対応値なし)
MP_FIELD_OF = {"a": "a", "c": "c", "c_over_a": "c_over_a",
               "volume_per_atom": "volume_per_atom", "density": "density",
               "gap": "band_gap", "b0": None, "magmom": None}


def _mp_key(mp_entry: Mapping[str, Any], key: str):
    field = MP_FIELD_OF.get(key)
    return mp_entry.get(field) if field else None


def summarise(rows: Sequence[dict], against: str = "exp") -> list[dict]:
    """物理量ごとの平均誤差・平均絶対誤差 (および相対値)。"""
    out: list[dict] = []
    for key, label, unit, _ in COMPARISONS:
        for group in (*GROUPS, "all"):
            selected = [r for r in rows
                        if r["quantity"] == key
                        and (group == "all" or r["group"] == group)
                        and r.get(f"err_{against}") is not None]
            if not selected:
                continue
            errors = np.array([r[f"err_{against}"] for r in selected], dtype=float)
            relative = np.array([r[f"relerr_{against}"] for r in selected], dtype=float)
            out.append({
                "quantity": key, "label": label, "unit": unit, "group": group,
                "n": len(selected),
                "me": float(errors.mean()), "mae": float(np.abs(errors).mean()),
                "mre": float(relative.mean()), "mare": float(np.abs(relative).mean()),
                "max_abs": float(np.abs(errors).max()),
            })
    return out


# -------------------------------------------------------------- レポート出力
def write_report(records: Sequence[dict], rows: Sequence[dict], root: str | Path,
                 backends: Sequence[str] = ("matplotlib",), dpi: int = 200) -> dict:
    """完了したベンチマークについて Markdown・CSV・パリティプロットを書き出す。"""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    # ---- CSV -------------------------------------------------------------
    columns = ["name", "group", "formula", "quantity", "unit", "calc", "exp", "mp",
               "err_exp", "relerr_exp", "err_mp", "relerr_mp", "mp_id", "mp_spacegroup", "mp_same_phase", "mp_excluded"]
    csv_path = root / "bench_comparison.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            fh.write(",".join(_csv(row.get(c)) for c in columns) + "\n")
    files["csv"] = str(csv_path)

    # ---- 図 ---------------------------------------------------------------
    plots: list[Path] = []
    for against in ("exp", "mp"):
        plots += parity_plots(rows, root / "plots", backends, dpi, against=against)
    files["plots"] = [str(p) for p in plots]

    # ---- Markdown ---------------------------------------------------------
    text = render_markdown(records, rows)
    report_path = root / "bench_report.md"
    report_path.write_text(text, encoding="utf-8")
    files["report"] = str(report_path)

    summary_path = root / "bench_summary.json"
    summary_path.write_text(json.dumps(
        {"n_systems": len(records),
         "n_ok": sum(1 for r in records if r.get("ok")),
         "summary_vs_experiment": summarise(rows, "exp"),
         "summary_vs_mp": summarise(rows, "mp"),
         "rows": list(rows)}, indent=2, default=str), encoding="utf-8")
    files["summary"] = str(summary_path)
    return files


def _csv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return f'"{text}"' if "," in text else text


#: 図の上で各参照値をどう呼ぶか
REFERENCE_LABEL = {"exp": ("実験値", "parity_vs_experiment"),
                   "mp": ("Materials Project", "parity_vs_mp")}


def parity_plots(rows: Sequence[dict], outdir: Path, backends: Sequence[str],
                 dpi: int = 200, against: str = "exp") -> list[Path]:
    """計算値と参照値の散布図。比較する物理量ごとに 1 パネル描く。"""
    from ezcal.plotting import resolve_backends

    backends = resolve_backends(list(backends))
    reference_name, stem = REFERENCE_LABEL[against]
    if not backends:
        return []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    key_of = f"err_{against}"
    panels = [(key, label, unit) for key, label, unit, _ in COMPARISONS
              if any(r["quantity"] == key and r.get(key_of) is not None for r in rows)]
    if not panels:
        return []

    colours = {"metals": "#1f4e9c", "oxides": "#b0392c"}
    if "matplotlib" in backends:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ncols = min(3, len(panels))
        nrows = math.ceil(len(panels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.8 * nrows),
                                 layout="constrained", squeeze=False)
        for axis, (key, label, unit) in zip(axes.flat, panels):
            selected = [r for r in rows
                        if r["quantity"] == key and r.get(key_of) is not None]
            for group in GROUPS:
                points = [r for r in selected if r["group"] == group]
                if not points:
                    continue
                axis.scatter([p[against] for p in points], [p["calc"] for p in points],
                             s=26, alpha=0.85, label=group, color=colours.get(group, "0.4"),
                             edgecolors="white", linewidths=0.5)
            values = [v for r in selected for v in (r[against], r["calc"])]
            lo, hi = min(values), max(values)
            pad = 0.06 * (hi - lo or 1.0)
            axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.55", lw=1, ls="--")
            axis.set_xlim(lo - pad, hi + pad)
            axis.set_ylim(lo - pad, hi + pad)
            axis.set_xlabel(f"{reference_name}{f'  ({unit})' if unit else ''}")
            axis.set_ylabel(f"ezcal / QE{f'  ({unit})' if unit else ''}")
            axis.set_title(f"{label}  (n={len(selected)})", fontsize=11)
            axis.grid(alpha=0.25)
        for axis in axes.flat[len(panels):]:
            axis.set_visible(False)
        axes.flat[0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"ezcal ベンチマーク: 計算値 vs {reference_name}", fontsize=12)
        path = outdir / f"{stem}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(path)

    if "plotly" in backends:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        ncols = min(3, len(panels))
        nrows = math.ceil(len(panels) / ncols)
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=[label for _, label, _ in panels])
        for index, (key, label, unit) in enumerate(panels):
            row, col = divmod(index, ncols)
            selected = [r for r in rows
                        if r["quantity"] == key and r.get(key_of) is not None]
            for group in GROUPS:
                points = [r for r in selected if r["group"] == group]
                if not points:
                    continue
                fig.add_trace(go.Scatter(
                    x=[p[against] for p in points], y=[p["calc"] for p in points],
                    mode="markers", name=group, legendgroup=group,
                    showlegend=index == 0, text=[p["name"] for p in points],
                    marker=dict(size=9, color=colours.get(group, "#666")),
                    hovertemplate=("%{text}<br>" + reference_name + " %{x:.3f}"
                                   "<br>ezcal %{y:.3f}<extra></extra>")),
                    row=row + 1, col=col + 1)
            values = [v for r in selected for v in (r[against], r["calc"])]
            lo, hi = min(values), max(values)
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(color="grey", dash="dash"),
                                     showlegend=False, hoverinfo="skip"),
                          row=row + 1, col=col + 1)
            fig.update_xaxes(title_text=unit or None, row=row + 1, col=col + 1)
        fig.update_layout(template="plotly_white", height=380 * nrows, width=380 * ncols,
                          title=f"ezcal ベンチマーク: 計算値 vs {reference_name}")
        path = outdir / f"{stem}.html"
        fig.write_html(path, include_plotlyjs="cdn")
        written.append(path)
    return written


def render_markdown(records: Sequence[dict], rows: Sequence[dict]) -> str:
    def fmt(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"

    ok = [r for r in records if r.get("ok")]
    failed = [r for r in records if not r.get("ok")]
    total_time = sum(float(r.get("elapsed_s") or 0) for r in records)

    lines = ["# ezcal ベンチマークレポート", "",
             f"- 系の数: **{len(records)}**  (成功 {len(ok)}、失敗 {len(failed)})",
             f"- 合計実時間: {total_time / 60:.1f} 分",
             "- パリティプロット: `plots/parity_vs_experiment.png` と "
             "`plots/parity_vs_mp.png` (対話版の `.html` も出力)", ""]

    for against, title in (("exp", "実験値との比較"), ("mp", "Materials Project との比較")):
        summary = summarise(rows, against)
        if not summary:
            continue
        lines += [f"## 平均誤差: {title}", "",
                  "| 物理量 | セット | n | ME | MAE | MRE (%) | MARE (%) | 最大絶対誤差 |",
                  "|---|---|---|---|---|---|---|---|"]
        for entry in summary:
            lines.append(
                f"| {entry['label']} ({entry['unit'] or '-'}) | {entry['group']} | {entry['n']} | "
                f"{entry['me']:+.4g} | {entry['mae']:.4g} | {entry['mre']:+.2f} | "
                f"{entry['mare']:.2f} | {entry['max_abs']:.4g} |")
        lines.append("")

    for group in GROUPS:
        subset = [r for r in records if r["group"] == group]
        if not subset:
            continue
        lines += [f"## {group}", "",
                  "| 系 | プロトタイプ | a 計算 | a 実験 | Δ% | V0/原子 | B0 計算 | B0 実験 | "
                  "ギャップ計算 | ギャップ実験 | m 計算 | m 実験 | 時間 (秒) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for record in subset:
            reference = record.get("reference") or {}
            relaxed = record.get("relaxed") or {}
            eos = record.get("eos") or {}
            calc_a = relaxed.get("a")
            exp_a = reference.get("a")
            delta = (100 * (calc_a - exp_a) / exp_a) if (calc_a and exp_a) else None
            moments = [abs(float(m)) for m in (record.get("site_magnetization") or [])
                       if abs(float(m)) > 0.05]
            lines.append(
                f"| {record['name']} | {record['prototype']} | {fmt(calc_a, 4)} | "
                f"{fmt(exp_a, 4)} | {fmt(delta, 2)} | "
                f"{fmt(relaxed.get('volume_per_atom'), 3)} | {fmt(eos.get('b0'), 1)} | "
                f"{fmt(reference.get('b0'), 1)} | {fmt(record.get('band_gap'), 3)} | "
                f"{fmt(reference.get('gap'), 2)} | {fmt(max(moments) if moments else None, 2)} | "
                f"{fmt(reference.get('magmom'), 2)} | {fmt(record.get('elapsed_s'), 0)} |")
        lines.append("")

    metal_dos = [r for r in records if (r.get("dos_metrics") or {}).get("n_ef") is not None]
    if metal_dos:
        lines += ["## フェルミ準位における状態密度", "",
                  "| 系 | N(E_F) (states/eV/cell) | 原子あたり | d バンド中心 (eV) |",
                  "|---|---|---|---|"]
        for record in metal_dos:
            metrics = record["dos_metrics"]
            lines.append(f"| {record['name']} | {fmt(metrics.get('n_ef'), 3)} | "
                         f"{fmt(metrics.get('n_ef_per_atom'), 3)} | "
                         f"{fmt(metrics.get('d_band_centre'), 3)} |")
        lines.append("")

    excluded = sorted({(r["name"], r.get("mp_id"), r.get("mp_spacegroup"))
                       for r in rows if r.get("mp_excluded")})
    if excluded:
        lines += ["## Materials Project: 相が異なるもの", "",
                  "以下の系について、MP はこちらが指定したプロトタイプとは異なる空間群へ"
                  "緩和している。そのため a / c / (c/a) は同じ物理量とは言えず、MP との"
                  "平均誤差からは除外している。原子あたり体積・密度・バンドギャップは"
                  "引き続き比較可能である。", "",
                  "| 系 | MP id | MP の空間群 |", "|---|---|---|"]
        for name, mp_id, spacegroup in excluded:
            lines.append(f"| {name} | {mp_id} | {spacegroup} |")
        lines.append("")

    lines += ["## 数値の読み方", "",
              "- 格子定数と体積弾性率は *構造* に対する検証である。PBE は格子定数を "
              "1 % 程度過大評価し、体積弾性率はおよそ 10 % 以内に収まる。ここで大きな"
              "誤差が出る場合は、擬ポテンシャル・カットオフ・k 点サンプリングの"
              "どれかに問題がある。",
              "- バンドギャップが実験値を大きく下回るのは **想定どおり** であり、"
              "PBE の既知の性質であってバグではない。ギャップについて意味があるのは "
              "Materials Project との比較のほうである (MP も PBE ベースのため)。",
              "- NiO / MnO / CoO は反強磁性として計算しているが **+U を入れていない**。"
              "そのためギャップも磁気モーメントも小さすぎる値になる。`--hubbard-u` を"
              "加えれば改善するが、その代わり素の PBE ではなくなる。",
              "- 凝集エネルギーと生成エネルギーは含めていない。孤立原子や O2 の参照計算と"
              "陰イオン補正が必要であり、環境が正常かを確かめるという目的とは別の"
              "作業になるためである。", ""]

    if failed:
        lines += ["## 失敗した系", ""]
        for record in failed:
            lines.append(f"- **{record['name']}**: {record.get('error') or '実行ディレクトリを参照'}")
            for message in (record.get("messages") or [])[-3:]:
                lines.append(f"  - {message}")
        lines.append("")
    return "\n".join(lines)
