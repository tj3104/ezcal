"""構造の入出力、対称性解析、k 点まわりの補助処理。

内部表現には :class:`pymatgen.core.Structure` を用いる。pymatgen が直接
読めない形式については ASE を利用する。
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MP_ID_RE = re.compile(r"^mp-\d+$|^mvc-\d+$")

#: Quantum ESPRESSO の元素ラベル ("Fe1", "Fe2", ...) を保持するサイトプロパティ
LABEL_PROP = "ezcal_label"
#: pw.x が元素ラベルとして受け付けるのは最大 3 文字
MAX_LABEL_LEN = 3


class StructureError(ValueError):
    pass


def label_element(label: str) -> str:
    """``"Fe1"`` -> ``"Fe"``、``"C_h"`` -> ``"C"``、``"Fe"`` -> ``"Fe"`` に正規化する。"""
    core = re.split(r"[_\-]", str(label))[0]
    return re.sub(r"\d+$", "", core) or str(label)


def site_labels(structure) -> list[str]:
    """サイトごとの元素ラベル。未設定なら素の元素記号を返す。"""
    stored = structure.site_properties.get(LABEL_PROP)
    if stored:
        return [str(value) for value in stored]
    return [site.specie.symbol for site in structure]


def set_site_labels(structure, labels: Sequence[str]):
    """``labels`` を元素ラベルとして付与した ``structure`` のコピーを返す。"""
    labels = [str(value) for value in labels]
    if len(labels) != len(structure):
        raise StructureError(
            f"サイト数 {len(structure)} に対してラベルが {len(labels)} 個あります")
    for label in labels:
        if len(label) > MAX_LABEL_LEN:
            raise StructureError(
                f"元素ラベル {label!r} が {MAX_LABEL_LEN} 文字を超えています。"
                "pw.x はこれを受け付けません")
    out = structure.copy()
    out.add_site_property(LABEL_PROP, labels)
    return out


def has_site_labels(structure) -> bool:
    """元素記号と異なるラベルを持つサイトが 1 つでもあれば True。"""
    return any(label != site.specie.symbol
               for label, site in zip(site_labels(structure), structure))


def split_sublattices(structure, spec: Mapping[str, Sequence[float]]):
    """元素を磁気副格子に分割する。

    ``spec`` は元素から各副格子の初期磁化への対応を表す。例:
    ``{"Fe": [0.6, -0.6]}``。該当元素のサイトには、構造中の出現順に
    ``Fe1``、``Fe2``、``Fe1``、... とラベルが振られる。これが反強磁性配置を
    可能にする仕組みで、pw.x は 2 つのラベルを 2 種類の元素として扱い、
    それに応じて対称性を下げる。

    戻り値は ``(ラベル付き構造, {ラベル: 磁化})``。
    """
    symbols = [site.specie.symbol for site in structure]
    labels = list(symbols)
    magnetization: dict[str, float] = {}

    for element, values in spec.items():
        values = [float(v) for v in (values if isinstance(values, (list, tuple)) else [values])]
        if not values:
            continue
        positions = [i for i, symbol in enumerate(symbols) if symbol == element]
        if not positions:
            raise StructureError(
                f"{element} はこの構造に含まれていません "
                f"(含まれるのは {', '.join(sorted(set(symbols)))})")
        if len(values) > 1 and len(positions) < len(values):
            raise StructureError(
                f"{element} に対して磁気副格子が {len(values)} 個指定されましたが、セル内の "
                f"{element} サイトは {len(positions)} 個しかありません。磁気秩序を収容できる大きさの "
                "磁気単位胞を用意し、縮約されないよう --as-is を付けてください。"
            )
        if len(values) == 1:
            magnetization[element] = values[0]
            continue
        for order, index in enumerate(positions):
            sub = order % len(values)
            label = f"{element}{sub + 1}"
            labels[index] = label
            magnetization[label] = values[sub]

    return set_site_labels(structure, labels), magnetization


# ------------------------------------------------------------------ 読み込み
def read_structure(source: str | Path, index: int = -1, api_key: str | None = None):
    """ファイルパスまたは Materials Project ID から構造を読み込む。

    対応形式: CIF、POSCAR/CONTCAR/*.vasp、*.xyz/extxyz、pymatgen の JSON、
    Quantum ESPRESSO の入出力、および ASE が読めるその他の形式。
    """
    from pymatgen.core import Structure

    text = str(source)
    if MP_ID_RE.match(text):
        return from_materials_project(text, api_key=api_key)

    path = Path(text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"構造が見つかりません: {path}  (mp-149 のような Materials Project ID も指定できます)"
        )

    suffix = path.suffix.lower()
    name = path.name.upper()

    if suffix == ".json":
        return Structure.from_file(str(path))
    if suffix == ".cif":
        return Structure.from_file(str(path), primitive=False)
    if suffix in {".vasp", ".poscar"} or name.startswith(("POSCAR", "CONTCAR")):
        return Structure.from_file(str(path))

    from ase.io import read as ase_read
    from pymatgen.io.ase import AseAtomsAdaptor

    fmt = None
    if suffix in {".in", ".pwi"}:
        fmt = "espresso-in"
    elif suffix in {".out", ".pwo"}:
        fmt = "espresso-out"
    atoms = ase_read(str(path), index=index, format=fmt)
    if isinstance(atoms, list):
        atoms = atoms[index]
    return AseAtomsAdaptor.get_structure(atoms)


def from_materials_project(mp_id: str, api_key: str | None = None):
    """Materials Project から構造をダウンロードする。"""
    import os

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "Materials Project の API キーが必要です: --mp-api-key を指定するか MP_API_KEY を設定してください"
        )
    try:
        from mp_api.client import MPRester
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mp-api がインストールされていません:  uv pip install mp-api") from exc

    with MPRester(key) as mpr:
        return mpr.get_structure_by_material_id(mp_id)


def write_structure(structure, path: str | Path, fmt: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path), fmt=fmt)
    return path


# --------------------------------------------------------------------- 変換
def to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def from_ase(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


# ------------------------------------------------------------------- 対称性
def standardize(structure, primitive: bool = True, symprec: float = 1e-5):
    """標準化した従来格子 (primitive=True ならプリミティブセル) を返す。"""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    if primitive:
        return sga.get_primitive_standard_structure()
    return sga.get_conventional_standard_structure()


def structure_info(structure, symprec: float = 1e-5) -> dict:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    info: dict[str, Any] = {
        "formula": structure.composition.reduced_formula,
        "full_formula": structure.composition.formula.replace(" ", ""),
        "natoms": len(structure),
        "nelements": len(structure.composition.elements),
        "elements": [str(el) for el in structure.composition.elements],
        "volume": round(float(structure.volume), 6),
        "density": round(float(structure.density), 6),
        "lattice": {
            "a": round(structure.lattice.a, 6),
            "b": round(structure.lattice.b, 6),
            "c": round(structure.lattice.c, 6),
            "alpha": round(structure.lattice.alpha, 4),
            "beta": round(structure.lattice.beta, 4),
            "gamma": round(structure.lattice.gamma, 4),
        },
    }
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        info["spacegroup"] = sga.get_space_group_symbol()
        info["spacegroup_number"] = sga.get_space_group_number()
        info["crystal_system"] = sga.get_crystal_system()
    except Exception:  # 対称性解析は失敗しても致命的ではない
        info["spacegroup"] = None
    return info


def is_metallic_guess(structure) -> bool:
    """占有数の既定値を選ぶためだけの、簡易的な判定。"""
    from pymatgen.core.periodic_table import Element

    return all(Element(str(el)).is_metal for el in structure.composition.elements)


# --------------------------------------------------------------------- k 点
def auto_kmesh(structure, kspacing: float = 0.25, min_points: int = 1) -> list[int]:
    """逆格子空間の間隔 (1/Angstrom) から Monkhorst-Pack メッシュを決める。

    ``kspacing`` は 2*pi の因子を含む。すなわち VASP の KSPACING タグと同じ
    流儀であり、0.25 はかなり密なメッシュにあたる。
    """
    recip = structure.lattice.reciprocal_lattice  # 既に 2*pi を含んでいる
    mesh = []
    for length in recip.abc:
        n = int(math.ceil(length / max(kspacing, 1e-6)))
        mesh.append(max(min_points, n))
    return mesh


def scale_kmesh(mesh: Sequence[int], factor: float) -> list[int]:
    return [max(1, int(math.ceil(n * factor))) for n in mesh]


@dataclass
class BandPath:
    """高対称 k 経路 (明示的な k 点列)。"""

    kpoints: np.ndarray                     # (nk, 3) 分率座標、プリミティブセル基準
    labels: list[tuple[int, str]] = field(default_factory=list)   # (インデックス, 表示ラベル)
    distances: np.ndarray | None = None     # (nk,) 累積の |k| (1/Angstrom)
    path: list[tuple[str, str]] = field(default_factory=list)
    primitive_structure: Any = None
    scheme: str = ""                        # 経路を決めた方式
    raw_labels: list[tuple[int, str]] = field(default_factory=list)  # 生ラベル (pymatgen 用)

    @property
    def nkpt(self) -> int:
        return len(self.kpoints)

    @property
    def reciprocal_lattice(self) -> np.ndarray | None:
        """プリミティブセルの逆格子 (行ベクトル、1/Angstrom、2*pi 込み)。"""
        if self.primitive_structure is None:
            return None
        return np.asarray(self.primitive_structure.lattice.reciprocal_lattice.matrix)


#: 選べるバンド経路の生成方式。既定は Materials Project と同じ方式。
BAND_PATH_SCHEMES = ("materials_project", "latimer_munro", "setyawan_curtarolo", "seekpath")

#: 既定の生成方式
DEFAULT_BAND_PATH_SCHEME = "materials_project"

_SCHEME_ALIASES = {
    "mp": "materials_project",
    "materials_project": "materials_project",
    "materialsproject": "materials_project",
    "materials-project": "materials_project",
    "latimer_munro": "latimer_munro",
    "latimer-munro": "latimer_munro",
    "lm": "latimer_munro",
    "setyawan_curtarolo": "setyawan_curtarolo",
    "setyawan-curtarolo": "setyawan_curtarolo",
    "sc": "setyawan_curtarolo",
    "seekpath": "seekpath",
    "hinuma": "seekpath",
    "hpkot": "seekpath",
}

#: pymatgen の HighSymmKpath へ渡す path_type
_PYMATGEN_PATH_TYPE = {"materials_project": "latimer_munro",
                       "latimer_munro": "latimer_munro",
                       "setyawan_curtarolo": "setyawan_curtarolo"}

#: 経路を一筆書き (オイラー路) に組み直す方式
_CONTINUOUS_SCHEMES = frozenset({"materials_project"})

#: 副格子ラベルを別種として対称性解析させるためのダミー元素 ("Xe" は実在元素なので除く)
_DUMMY_SYMBOLS = [f"X{c}" for c in "abcdfghijklmnopqrstuvwxyz"]


def resolve_band_scheme(scheme: str | None) -> str:
    """別名を正規の方式名へ直す。"""
    key = _SCHEME_ALIASES.get(str(scheme or DEFAULT_BAND_PATH_SCHEME).strip().lower())
    if key is None:
        raise StructureError(
            f"未知のバンド経路方式 {scheme!r} です。"
            f"使えるのは {', '.join(BAND_PATH_SCHEMES)} です")
    return key


_GREEK = {"GAMMA": "Γ", "DELTA": "Δ", "SIGMA": "Σ", "LAMBDA": "Λ"}


def _pretty_label(label: str) -> str:
    r"""経路ラベルを図に出せる形へ整える。

    pymatgen は TeX 記法 (``\Gamma``、``\Sigma_1``) を返す。seekpath の
    ``G`` は Gamma ではなく体心正方 (tI2) などで定義される別の点なので、
    Gamma として扱ってはいけない。
    """
    if not label:
        return ""
    label = str(label).lstrip("\\")
    base = label.replace("_", "")
    if base.upper() in _GREEK:
        return _GREEK[base.upper()]
    match = re.match(r"^([A-Za-z]+)_?(\d+)$", label)
    if match:
        head, digit = match.groups()
        head = _GREEK.get(head.upper(), head)
        return f"{head}$_{{{digit}}}$" if head not in _GREEK.values() else f"{head}{digit}"
    return label


def band_path(structure, line_density: float = 25.0, symprec: float = 1e-5,
              min_points: int = 6, scheme: str | None = None) -> BandPath:
    """高対称 k 経路を構築する。

    ``scheme`` は経路の決め方:

    ``materials_project`` (既定)
        pymatgen の Latimer-Munro 経路 (npj Comput. Mater. 6, 112 (2020)) を
        :meth:`HighSymmKpath.get_continuous_path` で一筆書き (オイラー路) に
        組み直したもの。Materials Project のバンド図と同じ横軸になり、経路に
        切れ目が生じない。
    ``latimer_munro``
        同じ Latimer-Munro 経路を、連続化せずそのまま使う。
    ``setyawan_curtarolo``
        pymatgen の Setyawan-Curtarolo (2010) 経路。古い文献や VASP 系の
        ツールでよく使われる規約。
    ``seekpath``
        seekpath の HPKOT 経路 (Hinuma 2017)。

    ``line_density`` は経路長 1/Angstrom あたりの k 点数 (逆格子は既に 2*pi を
    含んでいる)。Materials Project 本家は 20 を使う。

    ``materials_project`` 以外では経路に切れ目 (``U|K`` など) が残る。切れ目の
    両側は別の k 点なので、そこでバンドの値が飛ぶのは正しい挙動で、作図側で線を
    繋がないようにしている。
    """
    key = resolve_band_scheme(scheme)
    if key == "seekpath":
        prim, path, coords = _seekpath_kpath(structure, symprec)
    else:
        prim, path, coords = _pymatgen_kpath(structure, symprec, _PYMATGEN_PATH_TYPE[key])
    if key in _CONTINUOUS_SCHEMES:
        path = _eulerise_path(prim, path, coords)
    return _sample_path(prim, path, coords, line_density, min_points, key)


def _seekpath_kpath(structure, symprec: float):
    """seekpath (HPKOT) の経路と、それに揃えたプリミティブセル。"""
    import seekpath
    from pymatgen.core import Structure

    # 異なる元素ラベルは seekpath へ異なる型として渡す。こうすることで磁性セルは
    # 下がった対称性を保ち、正しいブリルアンゾーンが得られる
    labels = site_labels(structure)
    type_of = {label: index + 1 for index, label in enumerate(dict.fromkeys(labels))}
    label_of = {index: label for label, index in type_of.items()}

    cell = (
        structure.lattice.matrix.tolist(),
        structure.frac_coords.tolist(),
        [type_of[label] for label in labels],
    )
    res = seekpath.get_path(cell, with_time_reversal=True, symprec=symprec)

    prim_labels = [label_of[int(n)] for n in res["primitive_types"]]
    prim = Structure(
        lattice=res["primitive_lattice"],
        species=[label_element(label) for label in prim_labels],
        coords=res["primitive_positions"],
    )
    if any(label != element for label, element
           in zip(prim_labels, [site.specie.symbol for site in prim])):
        prim.add_site_property(LABEL_PROP, prim_labels)
    return prim, [tuple(seg) for seg in res["path"]], res["point_coords"]


def _pymatgen_kpath(structure, symprec: float, path_type: str):
    """pymatgen の経路と、その k 点座標が基準とする標準プリミティブセル。"""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from pymatgen.symmetry.bandstructure import HighSymmKpath

    # 副格子ラベルはダミー元素へ置き換える。そうしないと Fe1/Fe2 が同じ Fe と
    # みなされ、磁性セルの下がった対称性が拾えない
    decorated, label_of = _dummy_decorated(structure)
    sga = SpacegroupAnalyzer(decorated, symprec=symprec)
    # pymatgen の高対称点は標準プリミティブセルの逆格子基準で定義されている。
    # 先に自分で標準化しておき、以降の計算も同じセルで行う
    prim = sga.get_primitive_standard_structure(international_monoclinic=False)
    kpath = HighSymmKpath(prim, path_type=path_type, symprec=symprec)
    segments = [(group[i], group[i + 1])
                for group in kpath.kpath["path"]
                for i in range(len(group) - 1)]
    return _restore_labels(prim, label_of), segments, kpath.kpath["kpoints"]


def _connected_components(path):
    """区間を、端点を共有するかたまり (連結成分) ごとに分ける。"""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for start, end in path:
        a, b = find(start), find(end)
        if a != b:
            parent[a] = b

    groups: dict[str, list] = {}
    for segment in path:
        groups.setdefault(find(segment[0]), []).append(segment)
    return list(groups.values())


def _eulerise_component(prim, path, coords):
    """連結した区間の集まりを、pymatgen で一筆書き (オイラー路) に並べ替える。"""
    from pymatgen.core import Lattice
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.symmetry.bandstructure import HighSymmKpath

    recip = prim.lattice.reciprocal_lattice.matrix
    kpoints: list[np.ndarray] = []
    labels_dict: dict[str, list[float]] = {}
    for start, end in path:
        k0 = np.asarray(coords[start], dtype=float)
        k1 = np.asarray(coords[end], dtype=float)
        # 端点にだけラベルが付くよう、区間あたり 3 点 (端 - 中間 - 端) で足りる
        kpoints += [k0, k0 + 0.5 * (k1 - k0), k1]
        labels_dict.setdefault(str(start), k0.tolist())
        labels_dict.setdefault(str(end), k1.tolist())

    dummy = BandStructureSymmLine(
        np.asarray(kpoints), {Spin.up: np.zeros((1, len(kpoints)))},
        Lattice(recip), 0.0, labels_dict, coords_are_cartesian=False)
    continuous = HighSymmKpath.get_continuous_path(dummy)
    ordered = [(continuous.kpoints[branch["start_index"]].label,
                continuous.kpoints[branch["end_index"]].label)
               for branch in continuous.branches]
    if any(a not in coords or b not in coords for a, b in ordered):
        raise ValueError("一筆書きの経路に未知のラベルが含まれています")
    return ordered


def _eulerise_path(prim, path, coords):
    """区間を並べ替えて一筆書きにする (Materials Project と同じ扱い)。

    pymatgen の :meth:`HighSymmKpath.get_continuous_path` はグラフ理論で
    オイラー路を作る (奇数次数の頂点間に辺を足して一筆書きにする)。この関数は
    固有値を持たないダミーのバンド構造を通してその並べ替えだけを取り出す。
    追加された辺は同じ区間を 2 度通ることになるので、その分だけ k 点が増える。

    pymatgen の実装は経路全体が連結グラフでないと使えないが、磁気単位胞などでは
    孤立した区間が残ることがある。そこで連結成分ごとに掛け、成分の中は連続に、
    切れ目は成分の境目だけに抑える。個々の成分で失敗したらその成分は元の並びの
    まま残す。
    """
    ordered: list[tuple[str, str]] = []
    components = _connected_components(path)
    for component in components:
        try:
            ordered += _eulerise_component(prim, component, coords)
        except Exception as exc:                  # pragma: no cover - 経路依存
            warnings.warn(f"経路の一部を一筆書きにできませんでした ({exc})。"
                          "その部分は Latimer-Munro の並びのまま使います",
                          RuntimeWarning)
            ordered += list(component)
    if len(components) > 1:
        warnings.warn(
            f"バンド経路が {len(components)} 個に分かれています。"
            f"各かたまりの中は連続ですが、境目には切れ目が {len(components) - 1} "
            "個残ります", RuntimeWarning)
    return ordered


def _dummy_decorated(structure):
    """副格子ラベルを別種として扱わせるための構造を返す。

    戻り値は ``(構造, ダミー元素記号 -> 元のラベル)``。ラベルが素の元素記号の
    ままなら、構造をそのまま返す。
    """
    if not has_site_labels(structure):
        return structure, {}
    from pymatgen.core import Structure

    labels = site_labels(structure)
    order = list(dict.fromkeys(labels))
    if len(order) > len(_DUMMY_SYMBOLS):
        raise StructureError(
            f"副格子ラベルが {len(order)} 種類あります "
            f"(この経路方式で扱えるのは {len(_DUMMY_SYMBOLS)} 種類までです)")
    dummy_of = dict(zip(order, _DUMMY_SYMBOLS))
    decorated = Structure(structure.lattice, [dummy_of[label] for label in labels],
                          structure.frac_coords)
    return decorated, {symbol: label for label, symbol in dummy_of.items()}


def _restore_labels(prim, label_of: Mapping[str, str]):
    """ダミー元素で組んだセルを、元の元素と副格子ラベルへ戻す。"""
    if not label_of:
        return prim
    from pymatgen.core import Structure

    labels = [label_of[site.specie.symbol] for site in prim]
    out = Structure(prim.lattice, [label_element(label) for label in labels],
                    prim.frac_coords)
    if any(label != site.specie.symbol for label, site in zip(labels, out)):
        out.add_site_property(LABEL_PROP, labels)
    return out


def _sample_path(prim, path, coords, line_density: float, min_points: int,
                 scheme: str) -> BandPath:
    """区間ごとに k 点を並べ、累積経路長とラベルを付ける。"""
    recip = prim.lattice.reciprocal_lattice.matrix   # 行ベクトル、1/Angstrom (2*pi 込み)

    kpoints: list[list[float]] = []
    raw: list[tuple[int, str]] = []
    distances: list[float] = []
    total = 0.0

    for start, end in path:
        k0 = np.asarray(coords[start], dtype=float)
        k1 = np.asarray(coords[end], dtype=float)
        seg_len = float(np.linalg.norm((k1 - k0) @ recip))
        npts = max(min_points, int(round(seg_len * line_density)) + 1)

        # Materials Project / VASP の line mode と同じく、区間ごとに独立して
        # サンプリングする。区間のつなぎ目では同じ k 点が 2 度並ぶことになるが、
        # そのおかげで高対称点がすべて「区間の端」になり、pymatgen の
        # BandStructureSymmLine がブランチと目盛りを正しく組める。
        # 経路の切れ目 (U|K) では、同じ経路長に「異なる」k 点が 2 つ並ぶ。
        # つなぎ目か切れ目かは k 点座標を見れば区別でき、作図層がそこで線を切る。
        raw.append((len(kpoints), str(start)))
        for i in range(npts):
            frac = i / (npts - 1)
            kpoints.append((k0 + frac * (k1 - k0)).tolist())
            distances.append(total + frac * seg_len)
        total += seg_len
        raw.append((len(kpoints) - 1, str(end)))

    merged: dict[int, str] = {}
    merged_raw: dict[int, str] = {}
    for idx, lab in raw:
        merged[idx] = _merge_labels(merged.get(idx, ""), _pretty_label(lab))
        merged_raw[idx] = _merge_labels(merged_raw.get(idx, ""), lab)

    return BandPath(
        kpoints=np.asarray(kpoints, dtype=float),
        labels=sorted(merged.items()),
        distances=np.asarray(distances, dtype=float),
        path=[tuple(seg) for seg in path],
        primitive_structure=prim,
        scheme=scheme,
        raw_labels=sorted(merged_raw.items()),
    )


def _merge_labels(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b or a == b:
        return a
    return f"{a}|{b}"
