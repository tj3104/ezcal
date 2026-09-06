"""Charge density and atomic charges.

Three independent views of "where the electrons are", all from output ezcal
already produces:

``pp.x`` cube files
    the density itself, on the FFT grid.  ``plot_num=0`` is the pseudo
    (valence) density, ``6`` the spin density, ``17`` the all-electron
    valence density and ``21`` the full all-electron density - the last two
    are PAW reconstructions and are what a Bader partition wants.
Loewdin charges
    already present in the ``projwfc.x`` output the DOS step runs, so they
    cost nothing extra.  They also give a per-atom magnetic moment
    (``polarization``) independent of the sphere-integrated one pw.x prints.
Bader charges
    a grid partition of the density into atomic basins.  The on-grid
    algorithm of Henkelman, Arnaldsson and Jonsson (Comput. Mater. Sci. 36,
    354 (2006)) is implemented here so no external binary is needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

BOHR_ANG = 0.529177210903

#: pp.x plot_num values ezcal knows how to ask for
PLOT_NUM = {
    "density": 0,           # pseudo (valence) charge density
    "spin": 6,              # spin density, rho_up - rho_down
    "ae_valence": 17,       # all-electron valence density (PAW only)
    "ae_total": 21,         # all-electron density incl. core (PAW only)
    "potential": 11,        # bare + Hartree potential
}


class ChargeError(RuntimeError):
    pass


# ------------------------------------------------------------------- cube I/O
@dataclass
class CubeData:
    """A Gaussian cube file as written by ``pp.x``.

    ``density`` keeps the file's own units (e/bohr^3); lattice and positions
    are converted to Angstrom because that is what the rest of ezcal uses.
    """

    density: np.ndarray                  # (n1, n2, n3), e/bohr^3
    lattice: np.ndarray                  # (3, 3) Angstrom, rows are cell vectors
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
        """Cell volume in Angstrom^3."""
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def voxel_volume_bohr(self) -> float:
        return self.volume / BOHR_ANG ** 3 / self.density.size

    @property
    def electrons(self) -> float:
        """Integral of the density over the cell."""
        return float(self.density.sum() * self.voxel_volume_bohr)

    @property
    def symbols(self) -> list[str]:
        from pymatgen.core.periodic_table import Element

        return [Element.from_Z(int(z)).symbol for z in self.numbers]

    def fractional_positions(self) -> np.ndarray:
        return (self.positions - self.origin) @ np.linalg.inv(self.lattice)


def read_cube(path: str | Path) -> CubeData:
    """Read a Gaussian cube file (the format ``pp.x`` writes with output_format=6)."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.readline()
        handle.readline()                                  # two comment lines

        fields = handle.readline().split()
        natoms = int(fields[0])
        origin = np.array([float(v) for v in fields[1:4]]) * BOHR_ANG

        counts, vectors = [], []
        for _ in range(3):
            fields = handle.readline().split()
            counts.append(int(fields[0]))
            vectors.append([float(v) for v in fields[1:4]])
        # a negative count means the axis is in Angstrom already
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


# --------------------------------------------------------------- 1D / 2D views
def planar_average(cube: CubeData, axis: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Average over the two directions perpendicular to ``axis``.

    Returns (distance along the axis in Angstrom, average density).
    """
    others = tuple(i for i in range(3) if i != axis)
    average = cube.density.mean(axis=others)
    length = float(np.linalg.norm(cube.lattice[axis]))
    coords = np.linspace(0.0, length, cube.shape[axis], endpoint=False)
    return coords, average


def slice_plane(cube: CubeData, axis: int = 2,
                fraction: float = 0.5) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """A single grid plane through the cell, plus its extent in Angstrom."""
    n = cube.shape[axis]
    index = int(round(fraction * n)) % n
    data = np.take(cube.density, index, axis=axis)
    others = [i for i in range(3) if i != axis]
    extent = (0.0, float(np.linalg.norm(cube.lattice[others[1]])),
              0.0, float(np.linalg.norm(cube.lattice[others[0]])))
    return data, extent


# --------------------------------------------------------------- Bader basins
@dataclass
class BaderResult:
    electrons: np.ndarray                  # per atom
    volumes: np.ndarray                    # per atom, Angstrom^3
    charges: np.ndarray | None = None      # reference - electrons (positive = cation)
    reference: np.ndarray | None = None
    n_maxima: int = 0                      # basins holding a meaningful charge
    n_maxima_raw: int = 0                  # every local maximum on the grid
    max_offset: float = 0.0                # furthest significant maximum from its atom, A
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
    """Partition the density into atomic basins (on-grid Bader).

    Every grid point walks uphill to the steepest of its 26 neighbours, using
    the real-space step length so the ascent is not biased by the cell shape.
    Points that reach the same maximum form one basin, and each basin is
    given to the nearest atom.

    ``reference`` is the electron count each atom would have if neutral -
    ``Z_valence`` for a pseudo or all-electron *valence* density, the atomic
    number for a full all-electron density.  With it, ``charges`` becomes the
    transferred charge in units of e (positive = electrons removed).

    ``expected_electrons`` turns on the one check that catches a bad density:
    the integral over the cell has to come out at the electron count the
    calculation actually had.  A PAW reconstruction on too coarse a grid
    fails exactly here.
    """
    density = np.ascontiguousarray(cube.density, dtype=float)
    n1, n2, n3 = density.shape
    if cube.positions is None or len(cube.positions) == 0:
        raise ChargeError("the cube file carries no atomic positions")
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

    # pointer jumping: every point ends up pointing straight at its maximum
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
    delta -= np.round(delta)                                   # minimum image
    distances = np.linalg.norm(delta @ cube.lattice, axis=2)
    owner = distances.argmin(axis=1)
    offsets = distances[np.arange(len(maxima)), owner]

    atom_of_point = owner[np.searchsorted(maxima, roots)]
    voxel_volume = cube.volume / density.size
    electrons = np.bincount(atom_of_point, weights=density.ravel(),
                            minlength=natoms) * cube.voxel_volume_bohr
    volumes = np.bincount(atom_of_point, minlength=natoms) * voxel_volume

    # a smooth pseudo density has many shallow local maxima in the interstitial;
    # they are merged into the nearest atom and are not a problem.  Only basins
    # whose peak is a real feature of the density are worth checking.
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
                f"the density integrates to {result.total_electrons:.3f} e but the "
                f"calculation had {expected_electrons:.3f}: {drift:+.2f} %.  On a PAW "
                "reconstruction (plot_num 17/21) this means the FFT grid is too coarse "
                "for the cusps - use plot_num=0, or raise ecutrho")
    if result.max_offset > warn_offset:
        result.messages.append(
            f"a basin holding real charge peaks {result.max_offset:.2f} A from any atom - "
            "either a non-nuclear attractor or a grid that is too coarse")
    if reference is not None and abs(float(result.charges.sum())) > 0.05:
        result.messages.append(
            f"the basin charges sum to {result.charges.sum():+.3f} e instead of 0; "
            "check that the reference electron counts match the density that was used")
    return result


# ------------------------------------------------------------ Loewdin charges
_ATOM_RE = re.compile(r"Atom #\s*(\d+):\s*total charge\s*=\s*(-?[\d.]+)")
_SPIN_UP_RE = re.compile(r"spin up\s*=\s*(-?[\d.]+)")
_SPIN_DN_RE = re.compile(r"spin down\s*=\s*(-?[\d.]+)")
_POLAR_RE = re.compile(r"polarization\s*=\s*(-?[\d.]+)")
_ORBITAL_RE = re.compile(r",\s*([spdf])\s*=\s*(-?[\d.]+)")
_SPILL_RE = re.compile(r"Spilling Parameter:\s*(-?[\d.]+)")


def parse_lowdin(path: str | Path) -> dict:
    """Read the Loewdin charges out of a ``projwfc.x`` output file.

    projwfc.x already runs in the DOS step, so this costs nothing extra.  It
    yields a per-atom charge, its s/p/d/f breakdown and - for a spin
    polarised run - a per-atom magnetic moment (QE calls it ``polarization``).
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
            continue                                   # orbital-resolved moments, skip
        up, down = _SPIN_UP_RE.search(line), _SPIN_DN_RE.search(line)
        if up:
            entry["lowdin_up"] = float(up.group(1))
        if down:
            entry["lowdin_down"] = float(down.group(1))
        if not up and not down:                        # the "total charge" line
            for symbol, value in _ORBITAL_RE.findall(line):
                entry["orbitals"][symbol] = float(value)

    rows = [atoms[key] for key in sorted(atoms)]
    for row in rows:
        if "lowdin_moment" not in row and "lowdin_up" in row and "lowdin_down" in row:
            row["lowdin_moment"] = row["lowdin_up"] - row["lowdin_down"]
    return {"atoms": rows, "spilling": spilling}


# --------------------------------------------------------------- assembly
def atomic_charge_table(structure, lowdin: dict | None = None,
                        bader: BaderResult | None = None,
                        site_moments: Sequence[float] | None = None,
                        valence: Sequence[float] | None = None) -> list[dict]:
    """One row per atom, merging everything that is available."""
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
