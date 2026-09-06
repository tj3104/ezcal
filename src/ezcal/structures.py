"""Structure input/output, symmetry analysis and k-point helpers.

The internal representation is a :class:`pymatgen.core.Structure`; ASE is
used for the file formats pymatgen does not read natively.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

MP_ID_RE = re.compile(r"^mp-\d+$|^mvc-\d+$")

#: site property holding the Quantum ESPRESSO species label ("Fe1", "Fe2", ...)
LABEL_PROP = "ezcal_label"
#: pw.x accepts at most three characters for a species label
MAX_LABEL_LEN = 3


class StructureError(ValueError):
    pass


def label_element(label: str) -> str:
    """``"Fe1"`` -> ``"Fe"``, ``"C_h"`` -> ``"C"``, ``"Fe"`` -> ``"Fe"``."""
    core = re.split(r"[_\-]", str(label))[0]
    return re.sub(r"\d+$", "", core) or str(label)


def site_labels(structure) -> list[str]:
    """Per-site species labels; the plain element symbol when none were set."""
    stored = structure.site_properties.get(LABEL_PROP)
    if stored:
        return [str(value) for value in stored]
    return [site.specie.symbol for site in structure]


def set_site_labels(structure, labels: Sequence[str]):
    """Return a copy of ``structure`` carrying ``labels`` as species labels."""
    labels = [str(value) for value in labels]
    if len(labels) != len(structure):
        raise StructureError(
            f"got {len(labels)} labels for {len(structure)} sites")
    for label in labels:
        if len(label) > MAX_LABEL_LEN:
            raise StructureError(
                f"species label {label!r} is longer than {MAX_LABEL_LEN} characters, "
                "which pw.x does not accept")
    out = structure.copy()
    out.add_site_property(LABEL_PROP, labels)
    return out


def has_site_labels(structure) -> bool:
    """True when at least one site carries a label other than its element."""
    return any(label != site.specie.symbol
               for label, site in zip(site_labels(structure), structure))


def split_sublattices(structure, spec: Mapping[str, Sequence[float]]):
    """Split elements into magnetic sublattices.

    ``spec`` maps an element onto the starting magnetisations of its
    sublattices, e.g. ``{"Fe": [0.6, -0.6]}``.  The sites of that element
    are labelled ``Fe1``, ``Fe2``, ``Fe1``, ... in the order they appear in
    the structure, which is what makes an antiferromagnetic arrangement
    possible: pw.x treats two labels as two species and lowers the symmetry
    accordingly.

    Returns ``(labelled_structure, {label: magnetisation})``.
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
                f"{element} is not in this structure "
                f"(it contains {', '.join(sorted(set(symbols)))})")
        if len(values) > 1 and len(positions) < len(values):
            raise StructureError(
                f"{len(values)} magnetic sublattices were asked for {element} but the cell "
                f"has only {len(positions)} {element} site(s).  Supply a magnetic unit cell "
                "big enough to hold the ordering (and use --as-is so it is not reduced)."
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


# ----------------------------------------------------------------- reading
def read_structure(source: str | Path, index: int = -1, api_key: str | None = None):
    """Read a structure from a file path or a Materials Project id.

    Supported: CIF, POSCAR/CONTCAR/*.vasp, *.xyz/extxyz, pymatgen JSON,
    Quantum ESPRESSO input/output, and anything else ASE can read.
    """
    from pymatgen.core import Structure

    text = str(source)
    if MP_ID_RE.match(text):
        return from_materials_project(text, api_key=api_key)

    path = Path(text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"structure not found: {path}  (a Materials Project id such as mp-149 also works)"
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
    """Download a structure from the Materials Project."""
    import os

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError(
            "a Materials Project API key is required: pass --mp-api-key or set MP_API_KEY"
        )
    try:
        from mp_api.client import MPRester
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mp-api is not installed:  uv pip install mp-api") from exc

    with MPRester(key) as mpr:
        return mpr.get_structure_by_material_id(mp_id)


def write_structure(structure, path: str | Path, fmt: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path), fmt=fmt)
    return path


# ------------------------------------------------------------- conversions
def to_ase(structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_atoms(structure)


def from_ase(atoms):
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


# ---------------------------------------------------------------- symmetry
def standardize(structure, primitive: bool = True, symprec: float = 1e-5):
    """Return the (primitive) standardised conventional cell."""
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
    except Exception:  # symmetry analysis is best effort
        info["spacegroup"] = None
    return info


def is_metallic_guess(structure) -> bool:
    """Crude heuristic used only to pick a sensible default occupation scheme."""
    from pymatgen.core.periodic_table import Element

    return all(Element(str(el)).is_metal for el in structure.composition.elements)


# ---------------------------------------------------------------- k-points
def auto_kmesh(structure, kspacing: float = 0.25, min_points: int = 1) -> list[int]:
    """Monkhorst-Pack mesh from a reciprocal-space spacing in 1/Angstrom.

    ``kspacing`` includes the 2*pi factor, i.e. the same convention as
    VASP's KSPACING tag, so 0.25 is a fairly dense mesh.
    """
    recip = structure.lattice.reciprocal_lattice  # already contains 2*pi
    mesh = []
    for length in recip.abc:
        n = int(math.ceil(length / max(kspacing, 1e-6)))
        mesh.append(max(min_points, n))
    return mesh


def scale_kmesh(mesh: Sequence[int], factor: float) -> list[int]:
    return [max(1, int(math.ceil(n * factor))) for n in mesh]


@dataclass
class BandPath:
    """An explicit band path produced by seekpath."""

    kpoints: np.ndarray                     # (nk, 3) fractional, primitive cell
    labels: list[tuple[int, str]] = field(default_factory=list)   # (index, label)
    distances: np.ndarray | None = None     # (nk,) cumulative |k| in 1/Angstrom
    path: list[tuple[str, str]] = field(default_factory=list)
    primitive_structure: Any = None

    @property
    def nkpt(self) -> int:
        return len(self.kpoints)


_GREEK = {"GAMMA": "Γ", "G": "Γ", "DELTA": "Δ", "SIGMA": "Σ", "LAMBDA": "Λ"}


def _pretty_label(label: str) -> str:
    if not label:
        return ""
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
              min_points: int = 6) -> BandPath:
    """Build a high-symmetry k-path with seekpath.

    ``line_density`` is the number of k-points per 1/Angstrom of path
    length (reciprocal lattice already includes 2*pi).
    """
    import seekpath
    from pymatgen.core import Structure

    # distinct species labels are handed to seekpath as distinct types, so a
    # magnetic cell keeps its lowered symmetry and gets the right Brillouin zone
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
    recip = prim.lattice.reciprocal_lattice.matrix   # rows, 1/Angstrom incl. 2*pi
    coords = res["point_coords"]

    kpoints: list[list[float]] = []
    labels: list[tuple[int, str]] = []
    distances: list[float] = []
    total = 0.0

    for seg_idx, (start, end) in enumerate(res["path"]):
        k0 = np.asarray(coords[start], dtype=float)
        k1 = np.asarray(coords[end], dtype=float)
        seg_len = float(np.linalg.norm((k1 - k0) @ recip))
        npts = max(min_points, int(round(seg_len * line_density)) + 1)

        # a path break (U|K) puts two different k-points at the same path
        # length; both are kept and the plotting layer merges their labels
        continuous = bool(kpoints) and np.allclose(kpoints[-1], k0, atol=1e-8)
        if continuous:
            rng = range(1, npts)          # the previous segment ended here
        else:
            labels.append((len(kpoints), _pretty_label(start)))
            rng = range(0, npts)

        for i in rng:
            frac = i / (npts - 1)
            kpoints.append((k0 + frac * (k1 - k0)).tolist())
            distances.append(total + frac * seg_len)
        total += seg_len
        labels.append((len(kpoints) - 1, _pretty_label(end)))

    merged: dict[int, str] = {}
    for idx, lab in labels:
        merged[idx] = _merge_labels(merged.get(idx, ""), lab)

    return BandPath(
        kpoints=np.asarray(kpoints, dtype=float),
        labels=sorted(merged.items()),
        distances=np.asarray(distances, dtype=float),
        path=[tuple(p) for p in res["path"]],
        primitive_structure=prim,
    )


def _merge_labels(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a:
        return b
    if not b or a == b:
        return a
    return f"{a}|{b}"
