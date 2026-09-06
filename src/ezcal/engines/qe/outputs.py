"""Parsers for Quantum ESPRESSO output.

The XML schema written into ``<outdir>/<prefix>.save/data-file-schema.xml``
is the primary source (it is complete and machine readable); the text
output of ``pw.x`` is only used for run-time information and error
messages.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903
FORCE_HA_BOHR_TO_EV_ANG = HARTREE_EV / BOHR_ANG          # 51.42208
STRESS_HA_BOHR3_TO_GPA = 29421.015697                    # Ha/Bohr^3 -> GPa
RY_EV = 13.605693122994


def _floats(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(0)
    return np.fromstring(text.replace("D", "E").replace("d", "E"), sep=" ")


@dataclass
class PwXml:
    """Structured view of ``data-file-schema.xml``."""

    path: Path
    energy: float | None = None                  # eV
    fermi_energy: float | None = None            # eV
    homo: float | None = None
    lumo: float | None = None
    nelec: float | None = None
    nbnd: int | None = None
    lsda: bool = False
    kpoints_frac: np.ndarray | None = None       # (nk, 3)
    kweights: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None        # (nspin, nk, nbnd) in eV
    occupations: np.ndarray | None = None
    forces: np.ndarray | None = None             # (nat, 3) eV/Angstrom
    stress: np.ndarray | None = None             # (3, 3) GPa
    pressure: float | None = None                # GPa
    magnetization: float | None = None
    absolute_magnetization: float | None = None
    lattice: np.ndarray | None = None            # (3, 3) Angstrom
    positions_frac: np.ndarray | None = None
    symbols: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)   # pw.x species labels
    ecutwfc: float | None = None
    ecutrho: float | None = None
    functional: str | None = None
    converged: bool = False
    kmesh: list[int] | None = None
    spacegroup: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- derived --------------------------------------------------------
    @property
    def band_gap(self) -> float | None:
        if self.homo is None or self.lumo is None:
            return None
        return max(0.0, self.lumo - self.homo)

    def structure(self):
        from pymatgen.core import Lattice, Structure

        from ezcal.structures import LABEL_PROP

        if self.lattice is None or self.positions_frac is None:
            return None
        out = Structure(Lattice(self.lattice), self.symbols, self.positions_frac)
        if self.labels and any(a != b for a, b in zip(self.labels, self.symbols)):
            out.add_site_property(LABEL_PROP, list(self.labels))
        return out


def parse_xml(path: str | Path) -> PwXml:
    path = Path(path)
    root = ET.parse(path).getroot()
    out = root.find("output")
    if out is None:
        raise ValueError(f"{path}: no <output> section (did pw.x finish?)")
    res = PwXml(path=path)

    conv = root.find("output/convergence_info/scf_conv/convergence_achieved")
    if conv is None:
        conv = root.find("output/convergence_info/opt_conv/convergence_achieved")
    if conv is not None and conv.text:
        res.converged = conv.text.strip().lower() == "true"

    etot = out.findtext("total_energy/etot")
    if etot:
        res.energy = float(etot) * HARTREE_EV

    # -- structure -------------------------------------------------------
    struct = out.find("atomic_structure")
    if struct is not None:
        cell = struct.find("cell")
        if cell is not None:
            vectors = [ _floats(cell.findtext(tag)) for tag in ("a1", "a2", "a3") ]
            res.lattice = np.array(vectors) * BOHR_ANG
        positions = struct.find("atomic_positions")
        if positions is not None and res.lattice is not None:
            from ezcal.structures import label_element

            cart, symbols, labels = [], [], []
            for atom in positions.findall("atom"):
                cart.append(_floats(atom.text) * BOHR_ANG)
                raw = atom.get("name", "X")
                labels.append(raw)
                symbols.append(label_element(raw))
            if cart:
                cart_arr = np.array(cart)
                res.positions_frac = cart_arr @ np.linalg.inv(res.lattice)
                res.symbols = symbols
                res.labels = labels

    # -- basis / dft -----------------------------------------------------
    ecutwfc = out.findtext("basis_set/ecutwfc")
    if ecutwfc:
        res.ecutwfc = float(ecutwfc) * 2.0            # Hartree -> Ry
    ecutrho = out.findtext("basis_set/ecutrho")
    if ecutrho:
        res.ecutrho = float(ecutrho) * 2.0
    res.functional = out.findtext("dft/functional")

    # -- magnetisation ---------------------------------------------------
    total_mag = out.findtext("magnetization/total")
    if total_mag:
        res.magnetization = float(total_mag)
    abs_mag = out.findtext("magnetization/absolute")
    if abs_mag:
        res.absolute_magnetization = float(abs_mag)

    # -- forces / stress -------------------------------------------------
    forces = out.findtext("forces")
    if forces:
        arr = _floats(forces)
        if arr.size and arr.size % 3 == 0:
            res.forces = arr.reshape(-1, 3) * FORCE_HA_BOHR_TO_EV_ANG
    stress = out.findtext("stress")
    if stress:
        arr = _floats(stress)
        if arr.size == 9:
            res.stress = arr.reshape(3, 3) * STRESS_HA_BOHR3_TO_GPA
            res.pressure = float(np.trace(res.stress) / 3.0)

    # -- band structure --------------------------------------------------
    bs = out.find("band_structure")
    if bs is not None:
        res.lsda = (bs.findtext("lsda") or "false").strip().lower() == "true"
        nbnd_text = bs.findtext("nbnd") or bs.findtext("nbnd_up")
        res.nbnd = int(float(nbnd_text)) if nbnd_text else None
        nelec = bs.findtext("nelec")
        if nelec:
            res.nelec = float(nelec)
        fermi = bs.findtext("fermi_energy")
        if fermi:
            res.fermi_energy = float(fermi) * HARTREE_EV
        homo = bs.findtext("highestOccupiedLevel")
        if homo:
            res.homo = float(homo) * HARTREE_EV
        lumo = bs.findtext("lowestUnoccupiedLevel")
        if lumo:
            res.lumo = float(lumo) * HARTREE_EV

        mp = bs.find("starting_k_points/monkhorst_pack")
        if mp is not None:
            res.kmesh = [int(mp.get(f"nk{i}", 1)) for i in (1, 2, 3)]

        kpts, weights, eigs, occs = [], [], [], []
        alat = float(struct.get("alat")) if struct is not None and struct.get("alat") else None
        for entry in bs.findall("ks_energies"):
            kp = entry.find("k_point")
            if kp is None:
                continue
            kpts.append(_floats(kp.text))
            weights.append(float(kp.get("weight", 0.0)))
            eigs.append(_floats(entry.findtext("eigenvalues")) * HARTREE_EV)
            occs.append(_floats(entry.findtext("occupations")))
        if kpts:
            kcart = np.array(kpts)                     # units of 2*pi/alat
            if alat and res.lattice is not None:
                cell_bohr = res.lattice / BOHR_ANG
                res.kpoints_frac = kcart @ cell_bohr.T / alat
            else:
                res.kpoints_frac = kcart
            res.kweights = np.array(weights)
            eig = np.array(eigs)
            occ = np.array(occs)
            if res.lsda and res.nbnd:
                eig = eig.reshape(len(kpts), 2, res.nbnd).transpose(1, 0, 2)
                occ = occ.reshape(len(kpts), 2, res.nbnd).transpose(1, 0, 2)
            else:
                eig = eig[None, :, :]
                occ = occ[None, :, :]
            res.eigenvalues = eig
            res.occupations = occ
            if res.homo is None:
                res.homo, res.lumo = _homo_lumo(eig, occ)
    return res


def _homo_lumo(eig: np.ndarray, occ: np.ndarray, tol: float = 1e-4):
    occupied = eig[occ > tol]
    empty = eig[occ <= tol]
    homo = float(occupied.max()) if occupied.size else None
    lumo = float(empty.min()) if empty.size else None
    return homo, lumo


# ------------------------------------------------------------- text output
_WALL_RE = re.compile(r"PWSCF\s*:.*?([\d.]+)s\s+WALL", re.IGNORECASE)
_TIME_RE = re.compile(r"([\d.]+)m?\s*([\d.]+)?s\s+WALL")


_SITE_MAG_RE = re.compile(
    r"atom\s+(\d+)\s*\(R=[\d.]+\)\s+charge=\s*(-?[\d.]+)\s+magn=\s*(-?[\d.]+)")


#: recognisable pw.x failures -> what to actually do about them
ERROR_HINTS: tuple[tuple[str, str], ...] = (
    (r"S matrix not positive definite|routine cdiaghg",
     "the Davidson diagonalisation broke down: try --set dft.diagonalization=cg, or raise "
     "the cutoffs (this is common when ecutwfc is below the value the PAW/USPP "
     "pseudopotential asks for - 'ezcal pseudo fetch <element>' prints it)"),
    (r"convergence NOT achieved",
     "SCF did not converge: lower the mixing with --set dft.mixing_beta=0.2, try "
     "--set dft.mixing_mode=local-TF for a metal, or widen --degauss"),
    (r"too many bands are not converged",
     "raise --nbnd, or --set dft.electron_maxstep to a larger value"),
    (r"charge is wrong",
     "the charge density went bad: raise ecutrho, or start from a better initial "
     "magnetisation with --magmom"),
    (r"Not enough space allocated for radial FFT",
     "raise ecutrho (--ecutrho), the augmentation charge does not fit the grid"),
    (r"Maximum CPU time exceeded",
     "the job hit its time limit: raise run.timeout or the queue walltime"),
    (r"reading namelist|bad namelist|invalid.*namelist",
     "pw.x rejected the input file: check the values passed with --set"),
    (r"some processors have no G-vectors|too few G-vectors",
     "too many MPI ranks for this cell: lower --np"),
    (r"problems computing cholesky",
     "ill-conditioned overlap matrix: try --set dft.diagonalization=cg or raise the cutoffs"),
)


@dataclass
class PwText:
    converged: bool = False
    job_done: bool = False
    walltime: float | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    scf_energies: list[float] = field(default_factory=list)   # eV, one per scf step
    nsteps: int = 0
    site_magnetization: list[float] = field(default_factory=list)   # Bohr magneton
    total_magnetization: float | None = None
    absolute_magnetization: float | None = None


def parse_pw_text(path: str | Path) -> PwText:
    path = Path(path)
    res = PwText()
    if not path.is_file():
        return res
    text = path.read_text(encoding="utf-8", errors="replace")

    res.converged = "convergence has been achieved" in text
    res.job_done = "JOB DONE" in text
    res.scf_energies = [float(v) * RY_EV for v in
                        re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", text, re.M)]
    res.nsteps = len(re.findall(r"iteration #", text))

    # the last "Magnetic moment per site" block belongs to the converged state
    blocks = text.split("Magnetic moment per site")
    if len(blocks) > 1:
        res.site_magnetization = [float(m.group(3))
                                  for m in _SITE_MAG_RE.finditer(blocks[-1][:8000])]
    totals = re.findall(r"total magnetization\s+=\s+(-?[\d.]+)\s+Bohr", text)
    if totals:
        res.total_magnetization = float(totals[-1])
    absolutes = re.findall(r"absolute magnetization\s+=\s+(-?[\d.]+)\s+Bohr", text)
    if absolutes:
        res.absolute_magnetization = float(absolutes[-1])

    for match in re.finditer(r"%{5,}\s*\n(.*?)\n\s*%{5,}", text, re.S):
        block = " ".join(match.group(1).split())
        if block and block not in res.errors:
            res.errors.append(block[:500])
    if "Maximum CPU time exceeded" in text:
        res.errors.append("maximum CPU time exceeded")
    for pattern, message in (
        (r"convergence NOT achieved", "SCF convergence not achieved"),
        (r"too many bands are not converged", "too many bands are not converged"),
        (r"negative rho \(up, down\)", None),      # routine output, not a problem
    ):
        if message and re.search(pattern, text):
            res.warnings.append(message)

    # hints are matched against the reported problems, never the whole output:
    # "cdiaghg" also appears in the timing table of every healthy run
    haystack = " ".join(res.errors + res.warnings)
    for pattern, hint in ERROR_HINTS:
        if re.search(pattern, haystack, re.IGNORECASE) and hint not in res.hints:
            res.hints.append(hint)

    wall = re.findall(r"PWSCF\s*:.*?WALL", text)
    if wall:
        res.walltime = _parse_wall(wall[-1])
    return res


def _parse_wall(line: str) -> float | None:
    match = re.search(r"([\dhms.]+)\s+WALL", line)
    if not match:
        return None
    token = match.group(1)
    total, number = 0.0, ""
    for char in token:
        if char.isdigit() or char == ".":
            number += char
        elif char == "h":
            total += float(number or 0) * 3600
            number = ""
        elif char == "m":
            total += float(number or 0) * 60
            number = ""
        elif char == "s":
            total += float(number or 0)
            number = ""
    if number:
        total += float(number)
    return total or None


# ---------------------------------------------------------------- dos.x
def read_dos(path: str | Path) -> dict:
    """Read a ``dos.x`` output file (E, dos, integrated dos)."""
    path = Path(path)
    fermi = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    match = re.search(r"EFermi\s*=\s*(-?[\d.]+)", header)
    if match:
        fermi = float(match.group(1))
    data = np.loadtxt(path, comments="#")
    out = {"energy": data[:, 0], "fermi_energy": fermi}
    if data.shape[1] >= 4:              # spin polarised: E, dosup, dosdw, idos
        out["dos_up"] = data[:, 1]
        out["dos_down"] = data[:, 2]
        out["dos"] = data[:, 1] + data[:, 2]
        out["idos"] = data[:, 3]
    else:
        out["dos"] = data[:, 1]
        out["idos"] = data[:, 2] if data.shape[1] > 2 else None
    return out


# --------------------------------------------------------------- projwfc
# the species field can be a sublattice label ("Ni1"), not just an element
_PDOS_RE = re.compile(
    r"pdos_atm#(\d+)\(([A-Za-z][A-Za-z0-9_\-]*)\)_wfc#(\d+)\(([a-z]).*\)")


def read_pdos(workdir: str | Path, filpdos: str = "pdos") -> dict:
    """Collect ``projwfc.x`` output into per element / per orbital channels.

    For a spin polarised run the up and down channels are kept apart
    (``per_orbital_up`` / ``per_orbital_down``) as well as summed.
    """
    workdir = Path(workdir)
    files = sorted(workdir.glob(f"{filpdos}.pdos_atm#*"))
    if not files:
        return {}

    energy = None
    sums: dict[str, dict[str, np.ndarray]] = {
        "per_element": {}, "per_orbital": {},
        "per_element_up": {}, "per_element_down": {},
        "per_orbital_up": {}, "per_orbital_down": {},
    }
    spin_polarised = False

    for path in files:
        match = _PDOS_RE.search(path.name)
        if not match:
            continue
        _, element, _, orbital = match.groups()
        data = np.loadtxt(path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        if energy is None:
            energy = data[:, 0]
        orbital_key = f"{element}-{orbital}"

        if data.shape[1] >= 3 and _is_spin_file(path):
            spin_polarised = True
            up, down = data[:, 1], data[:, 2]
            for key, target in ((element, "per_element_up"), (orbital_key, "per_orbital_up")):
                sums[target][key] = sums[target].get(key, 0) + up
            for key, target in ((element, "per_element_down"),
                                (orbital_key, "per_orbital_down")):
                sums[target][key] = sums[target].get(key, 0) + down
            total = up + down
        else:
            total = data[:, 1]
        sums["per_element"][element] = sums["per_element"].get(element, 0) + total
        sums["per_orbital"][orbital_key] = sums["per_orbital"].get(orbital_key, 0) + total

    total_path = workdir / f"{filpdos}.pdos_tot"
    total_dos = None
    if total_path.is_file():
        data = np.loadtxt(total_path, comments="#")
        total_dos = data[:, 1] if data.shape[1] < 4 else data[:, 1] + data[:, 2]

    out: dict = {"energy": energy, "total": total_dos, "spin_polarised": spin_polarised}
    for key, values in sums.items():
        out[key] = {k: np.asarray(v) for k, v in values.items()}
    return out


def _is_spin_file(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    return "ldosup" in header or "dosup" in header
