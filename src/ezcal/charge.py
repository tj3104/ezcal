"""電荷密度と原子電荷。

「電子がどこにいるか」を、ezcal が既に生成している出力だけから 3 通りの
独立した見方で示す:

``pp.x`` の cube ファイル
    FFT グリッド上の電荷密度そのもの。``plot_num=0`` は擬ポテンシャルの
    価電子密度、``6`` はスピン密度、``17`` は全電子の価電子密度、``21`` は
    内殻を含む全電子密度。後ろの 2 つは PAW 再構成であり、Bader 分割に
    適しているのはこちら。
Loewdin 電荷
    DOS ステップで実行される ``projwfc.x`` の出力に既に含まれているため、
    追加コストはゼロ。pw.x が出力する球積分値とは独立に、原子ごとの磁気
    モーメント (``polarization``) も得られる。
Bader 電荷
    電荷密度をグリッド上で原子ベイスンに分割したもの。Henkelman, Arnaldsson,
    Jonsson の on-grid アルゴリズム (Comput. Mater. Sci. 36, 354 (2006)) を
    ここで実装しているため、外部バイナリは不要。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BOHR_ANG = 0.529177210903

#: ezcal が指定方法を把握している pp.x の plot_num
PLOT_NUM = {
    "density": 0,           # 擬ポテンシャルの価電子密度
    "spin": 6,              # スピン密度、rho_up - rho_down
    "ae_valence": 17,       # 全電子の価電子密度 (PAW のみ)
    "ae_total": 21,         # 内殻を含む全電子密度 (PAW のみ)
    "potential": 11,        # 裸のポテンシャル + ハートリーポテンシャル
}


class ChargeError(RuntimeError):
    pass


# ------------------------------------------------------------------ cube I/O
@dataclass
class CubeData:
    """``pp.x`` が書き出す Gaussian cube ファイル。

    ``density`` はファイル本来の単位 (e/bohr^3) のまま保持する。格子と原子位置は
    ezcal の他の部分に合わせて Angstrom に変換する。
    """

    density: np.ndarray                  # (n1, n2, n3)、e/bohr^3
    lattice: np.ndarray                  # (3, 3) Angstrom、各行がセルベクトル
    origin: np.ndarray                   # (3,) Angstrom
    numbers: list[int] = field(default_factory=list)
    positions: np.ndarray | None = None  # (nat, 3) Angstrom
    path: Path | None = None
    kind: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) for n in self.density.shape)

    @property
    def volume(self) -> float:
        """セルの体積 (Angstrom^3)。"""
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def voxel_volume_bohr(self) -> float:
        return self.volume / BOHR_ANG ** 3 / self.density.size

    @property
    def electrons(self) -> float:
        """セル全体にわたる電荷密度の積分値。"""
        return float(self.density.sum() * self.voxel_volume_bohr)

    @property
    def symbols(self) -> list[str]:
        from pymatgen.core.periodic_table import Element

        return [Element.from_Z(int(z)).symbol for z in self.numbers]

    def fractional_positions(self) -> np.ndarray:
        return (self.positions - self.origin) @ np.linalg.inv(self.lattice)


def read_cube(path: str | Path) -> CubeData:
    """Gaussian cube ファイルを読む (``pp.x`` の output_format=6 で出力される形式)。"""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.readline()
        handle.readline()                                  # コメント行が 2 行

        fields = handle.readline().split()
        natoms = int(fields[0])
        origin = np.array([float(v) for v in fields[1:4]]) * BOHR_ANG

        counts, vectors = [], []
        for _ in range(3):
            fields = handle.readline().split()
            counts.append(int(fields[0]))
            vectors.append([float(v) for v in fields[1:4]])
        # 点数が負の場合、その軸は既に Angstrom 単位
        voxel = np.array(vectors) * np.array(
            [BOHR_ANG if n > 0 else 1.0 for n in counts])[:, None]
        counts = [abs(n) for n in counts]
        lattice = voxel * np.array(counts)[:, None]

        numbers, positions = [], []
        for _ in range(abs(natoms)):
            fields = handle.readline().split()
            numbers.append(int(fields[0]))
            positions.append([float(v) for v in fields[2:5]])
        positions = np.array(positions) * BOHR_ANG if positions else np.zeros((0, 3))

        values = np.fromstring(handle.read().replace("\n", " "), sep=" ")

    expected = counts[0] * counts[1] * counts[2]
    if values.size < expected:
        raise ChargeError(f"{path}: expected {expected} grid values, found {values.size}")
    density = values[:expected].reshape(counts)
    return CubeData(density=density, lattice=lattice, origin=origin,
                    numbers=numbers, positions=positions, path=path)


# -------------------------------------------------------------- 1D / 2D 表示
def planar_average(cube: CubeData, axis: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``axis`` に垂直な 2 方向について平均を取る。

    戻り値は (軸に沿った距離 (Angstrom), 平均密度)。
    """
    others = tuple(i for i in range(3) if i != axis)
    average = cube.density.mean(axis=others)
    length = float(np.linalg.norm(cube.lattice[axis]))
    coords = np.linspace(0.0, length, cube.shape[axis], endpoint=False)
    return coords, average


def slice_plane(cube: CubeData, axis: int = 2,
                fraction: float = 0.5) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """セルを横切るグリッド 1 面と、その広がり (Angstrom)。"""
    n = cube.shape[axis]
    index = int(round(fraction * n)) % n
    data = np.take(cube.density, index, axis=axis)
    others = [i for i in range(3) if i != axis]
    extent = (0.0, float(np.linalg.norm(cube.lattice[others[1]])),
              0.0, float(np.linalg.norm(cube.lattice[others[0]])))
    return data, extent


# ------------------------------------------------------------- Bader ベイスン
@dataclass
class BaderResult:
    electrons: np.ndarray                  # 原子ごと
    volumes: np.ndarray                    # 原子ごと、Angstrom^3
    charges: np.ndarray | None = None      # 基準値 - 電子数 (正なら陽イオン)
    reference: np.ndarray | None = None
    n_maxima: int = 0                      # 意味のある電荷を持つベイスンの数
    n_maxima_raw: int = 0                  # グリッド上の極大点すべて
    max_offset: float = 0.0                # 有意な極大点と担当原子との最大距離 (A)
    total_electrons: float = 0.0
    grid: tuple[int, int, int] = (0, 0, 0)
    messages: list[str] = field(default_factory=list)

    def as_rows(self, symbols: Sequence[str]) -> list[dict]:
        rows = []
        for index, symbol in enumerate(symbols):
            row = {"index": index + 1, "symbol": symbol,
                   "bader_electrons": float(self.electrons[index]),
                   "bader_volume": float(self.volumes[index])}
            if self.charges is not None:
                row["bader_charge"] = float(self.charges[index])
            rows.append(row)
        return rows


def bader_charges(cube: CubeData, reference: Sequence[float] | None = None,
                  warn_offset: float = 0.6, significant: float = 0.01,
                  expected_electrons: float | None = None) -> BaderResult:
    """電荷密度を原子ベイスンに分割する (on-grid Bader 法)。

    各グリッド点は 26 近傍のうち最も急な方向へ登っていく。その際に実空間での
    ステップ長を使うため、セル形状によって登り方が偏ることはない。同じ極大点に
    到達した点が 1 つのベイスンを構成し、各ベイスンは最も近い原子に割り当てられる。

    ``reference`` は各原子が中性であるときの電子数。擬ポテンシャルまたは全電子の
    *価電子* 密度なら ``Z_valence``、内殻を含む全電子密度なら原子番号を渡す。これを
    与えると ``charges`` が電荷移動量 (単位 e、正なら電子が奪われた側) になる。

    ``expected_electrons`` を渡すと、密度が破綻していないかを判定できる唯一の
    チェックが有効になる。すなわち、セル全体の積分値が実際の計算の電子数と一致
    しなければならない。粗すぎるグリッド上の PAW 再構成はまさにここで失敗する。
    """
    density = np.ascontiguousarray(cube.density, dtype=float)
    n1, n2, n3 = density.shape
    if cube.positions is None or len(cube.positions) == 0:
        raise ChargeError("この cube ファイルには原子位置が含まれていません")
    natoms = len(cube.positions)

    voxel = cube.lattice / np.array([n1, n2, n3], dtype=float)[:, None]
    index = np.arange(density.size, dtype=np.int64).reshape(density.shape)
    best_slope = np.zeros(density.shape)
    parent = index.copy()

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                step = np.array([di, dj, dk], dtype=float) @ voxel
                distance = float(np.linalg.norm(step))
                shift = (-di, -dj, -dk)
                slope = (np.roll(density, shift, axis=(0, 1, 2)) - density) / distance
                better = slope > best_slope
                if not better.any():
                    continue
                best_slope = np.where(better, slope, best_slope)
                parent = np.where(better, np.roll(index, shift, axis=(0, 1, 2)), parent)

    # ポインタジャンプ: 最終的にすべての点が自分の極大点を直接指すようにする
    flat = parent.ravel()
    for _ in range(64):
        nxt = flat[flat]
        if np.array_equal(nxt, flat):
            break
        flat = nxt
    roots = flat

    maxima = np.unique(roots)
    grid_i, grid_j, grid_k = np.unravel_index(maxima, density.shape)
    maxima_frac = np.stack([grid_i / n1, grid_j / n2, grid_k / n3], axis=1)

    atom_frac = cube.fractional_positions()
    delta = maxima_frac[:, None, :] - atom_frac[None, :, :]
    delta -= np.round(delta)                                   # 最小イメージ規約
    distances = np.linalg.norm(delta @ cube.lattice, axis=2)
    owner = distances.argmin(axis=1)
    offsets = distances[np.arange(len(maxima)), owner]

    atom_of_point = owner[np.searchsorted(maxima, roots)]
    voxel_volume = cube.volume / density.size
    electrons = np.bincount(atom_of_point, weights=density.ravel(),
                            minlength=natoms) * cube.voxel_volume_bohr
    volumes = np.bincount(atom_of_point, minlength=natoms) * voxel_volume

    # 滑らかな擬電荷密度は格子間領域に浅い極大点を多数持つ。これらは最寄りの原子に
    # 併合されるので問題にならない。検査する価値があるのは、ピークが密度の実体的な
    # 特徴となっているベイスンだけである。
    peak = density.ravel()[maxima]
    strong = peak > significant * float(density.max())

    result = BaderResult(
        electrons=electrons, volumes=volumes,
        n_maxima=int(strong.sum()), n_maxima_raw=int(len(maxima)),
        max_offset=float(offsets[strong].max()) if strong.any() else 0.0,
        total_electrons=float(electrons.sum()), grid=(n1, n2, n3))
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        result.reference = reference
        result.charges = reference - electrons

    if expected_electrons:
        drift = 100.0 * (result.total_electrons - expected_electrons) / expected_electrons
        if abs(drift) > 0.5:
            result.messages.append(
                f"電荷密度の積分値が {result.total_electrons:.3f} e ですが、計算の電子数は "
                f"{expected_electrons:.3f} です ({drift:+.2f} %)。PAW 再構成 "
                "(plot_num 17/21) の場合、これは尖点に対して FFT グリッドが粗すぎることを"
                "意味します。plot_num=0 を使うか、ecutrho を上げてください")
    if result.max_offset > warn_offset:
        result.messages.append(
            f"実体的な電荷を持つベイスンのピークが、どの原子からも {result.max_offset:.2f} A "
            "離れています。非核アトラクタか、グリッドが粗すぎるかのどちらかです")
    if reference is not None and abs(float(result.charges.sum())) > 0.05:
        result.messages.append(
            f"ベイスン電荷の総和が 0 ではなく {result.charges.sum():+.3f} e になっています。"
            "基準電子数が、使用した電荷密度と整合しているか確認してください")
    return result


# ----------------------------------------------------------- Loewdin 電荷
_ATOM_RE = re.compile(r"Atom #\s*(\d+):\s*total charge\s*=\s*(-?[\d.]+)")
_SPIN_UP_RE = re.compile(r"spin up\s*=\s*(-?[\d.]+)")
_SPIN_DN_RE = re.compile(r"spin down\s*=\s*(-?[\d.]+)")
_POLAR_RE = re.compile(r"polarization\s*=\s*(-?[\d.]+)")
_ORBITAL_RE = re.compile(r",\s*([spdf])\s*=\s*(-?[\d.]+)")
_SPILL_RE = re.compile(r"Spilling Parameter:\s*(-?[\d.]+)")


def parse_lowdin(path: str | Path) -> dict:
    """``projwfc.x`` の出力ファイルから Loewdin 電荷を読み取る。

    projwfc.x は DOS ステップで既に実行されているため、追加コストはかからない。
    原子ごとの電荷、その s/p/d/f 分解、そしてスピン分極計算であれば原子ごとの
    磁気モーメント (QE の言う ``polarization``) が得られる。
    """
    path = Path(path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("Lowdin Charges")
    if start < 0:
        return {}
    block = text[start:]
    end = block.find("Spilling Parameter")
    spilling = None
    if end >= 0:
        match = _SPILL_RE.search(block[end:])
        if match:
            spilling = float(match.group(1))
        block = block[:end]

    atoms: dict[int, dict[str, Any]] = {}
    current: int | None = None
    for line in block.splitlines():
        match = _ATOM_RE.search(line)
        if match:
            current = int(match.group(1))
            atoms.setdefault(current, {"index": current,
                                       "lowdin_electrons": float(match.group(2)),
                                       "orbitals": {}})
        if current is None:
            continue
        entry = atoms[current]
        if "polarization" in line:
            polar = _POLAR_RE.search(line)
            if polar:
                entry["lowdin_moment"] = float(polar.group(1))
            continue                                   # 軌道分解のモーメント行は読み飛ばす
        up, down = _SPIN_UP_RE.search(line), _SPIN_DN_RE.search(line)
        if up:
            entry["lowdin_up"] = float(up.group(1))
        if down:
            entry["lowdin_down"] = float(down.group(1))
        if not up and not down:                        # "total charge" の行
            for symbol, value in _ORBITAL_RE.findall(line):
                entry["orbitals"][symbol] = float(value)

    rows = [atoms[key] for key in sorted(atoms)]
    for row in rows:
        if "lowdin_moment" not in row and "lowdin_up" in row and "lowdin_down" in row:
            row["lowdin_moment"] = row["lowdin_up"] - row["lowdin_down"]
    return {"atoms": rows, "spilling": spilling}


# ---------------------------------------------------------------- 集約処理
def atomic_charge_table(structure, lowdin: dict | None = None,
                        bader: BaderResult | None = None,
                        site_moments: Sequence[float] | None = None,
                        valence: Sequence[float] | None = None) -> list[dict]:
    """利用できる情報をすべて統合し、原子ごとに 1 行を作る。"""
    from ezcal.structures import site_labels

    labels = site_labels(structure)
    lowdin_rows = (lowdin or {}).get("atoms") or []
    rows: list[dict] = []
    for index, site in enumerate(structure):
        row: dict[str, Any] = {
            "index": index + 1,
            "label": labels[index],
            "element": site.specie.symbol,
            "x": round(float(site.coords[0]), 6),
            "y": round(float(site.coords[1]), 6),
            "z": round(float(site.coords[2]), 6),
        }
        if valence is not None and index < len(valence):
            row["z_valence"] = float(valence[index])
        if index < len(lowdin_rows):
            entry = lowdin_rows[index]
            row["lowdin_electrons"] = entry.get("lowdin_electrons")
            if valence is not None and index < len(valence) and entry.get("lowdin_electrons"):
                row["lowdin_charge"] = float(valence[index]) - entry["lowdin_electrons"]
            if entry.get("lowdin_moment") is not None:
                row["lowdin_moment"] = entry["lowdin_moment"]
            for symbol, value in (entry.get("orbitals") or {}).items():
                row[f"lowdin_{symbol}"] = value
        if bader is not None and index < len(bader.electrons):
            row["bader_electrons"] = float(bader.electrons[index])
            row["bader_volume"] = float(bader.volumes[index])
            if bader.charges is not None:
                row["bader_charge"] = float(bader.charges[index])
        if site_moments is not None and index < len(site_moments):
            row["moment_sphere"] = float(site_moments[index])
        rows.append(row)
    return rows


def write_charge_csv(rows: Sequence[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(
                "" if row.get(c) is None else
                (f"{row[c]:.6g}" if isinstance(row[c], float) else str(row[c]))
                for c in columns) + "\n")
    return path
