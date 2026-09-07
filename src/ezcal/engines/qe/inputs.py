"""Quantum ESPRESSO の入力ファイル生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

BOHR = 0.529177210903

#: 各元素で使う Hubbard 多様体の指定 (QE 7.1 以降の HUBBARD カード)
_BLOCK_OFFSET = {"s": 0, "p": 0, "d": -1, "f": -2}


def fortran(value: Any) -> str:
    """Python の値を Fortran の namelist 記法で書き出す。"""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, float):
        text = f"{value:.10g}"
        return text.replace("e", "d") if "e" in text else text
    return str(value)


def namelist(name: str, values: Mapping[str, Any]) -> str:
    lines = [f"&{name.upper()}"]
    for key, value in values.items():
        if value is None:
            continue
        lines.append(f"  {key} = {fortran(value)}")
    lines.append("/")
    return "\n".join(lines)


def hubbard_manifold(label: str) -> str:
    """``Fe`` -> ``Fe-3d``。副格子ラベルはその名前のまま (``Fe1-3d``)。"""
    from pymatgen.core.periodic_table import Element

    from ezcal.structures import label_element

    el = Element(label_element(label))
    block = el.block
    n = el.row + _BLOCK_OFFSET.get(block, 0)
    return f"{label}-{n}{block}"


def species_list(structure) -> list[str]:
    """pw.x の元素種を、最初に現れた順に重複なく並べる。

    磁気副格子がある場合、ここに入るのは素の元素記号ではなくラベル
    (``Fe1``、``Fe2``) になる。
    """
    from ezcal.structures import site_labels

    seen: list[str] = []
    for label in site_labels(structure):
        if label not in seen:
            seen.append(label)
    return seen


class PwInput:
    """構造と ezcal の設定から ``pw.x`` の入力ファイルを組み立てる。"""

    def __init__(
        self,
        structure,
        pseudos: Mapping[str, Any],
        config,
        calculation: str = "scf",
        prefix: str = "ezcal",
        outdir: str = "./tmp",
        pseudo_dir: str | Path = ".",
        kmesh: Sequence[int] | None = None,
        kpoints_explicit: np.ndarray | None = None,
        nbnd: int | None = None,
        occupations: str | None = None,
        extra: Mapping[str, Mapping[str, Any]] | None = None,
        startingpot: str | None = None,
        startingwfc: str | None = None,
    ) -> None:
        self.structure = structure
        self.pseudos = pseudos
        self.config = config
        self.calculation = calculation
        self.prefix = prefix
        self.outdir = str(outdir)
        self.pseudo_dir = str(pseudo_dir)
        self.kmesh = list(kmesh) if kmesh is not None else None
        self.kpoints_explicit = kpoints_explicit
        self.nbnd = nbnd
        self.occupations = occupations
        self.extra = {k: dict(v) for k, v in (extra or {}).items()}
        self.startingpot = startingpot
        self.startingwfc = startingwfc
        self.species = species_list(structure)

    # ------------------------------------------------------------ 各ブロック
    def control(self) -> dict:
        cfg = self.config
        calc = "vc-relax" if self.calculation == "vc-relax" else self.calculation
        values: dict[str, Any] = {
            "calculation": calc,
            "prefix": self.prefix,
            "outdir": self.outdir,
            "pseudo_dir": self.pseudo_dir,
            "verbosity": "high",
            "tprnfor": bool(cfg.get("dft.tprnfor", True)),
            "tstress": bool(cfg.get("dft.tstress", True)),
            "disk_io": "low" if calc in {"scf", "relax", "vc-relax"} else "low",
        }
        if calc in {"relax", "vc-relax"}:
            values["etot_conv_thr"] = float(cfg.get("relax.etot_conv_thr", 1e-5))
            values["forc_conv_thr"] = float(cfg.get("relax.forc_conv_thr", 1e-4))
            values["nstep"] = int(cfg.get("relax.nstep", 100))
        if calc in {"nscf", "bands"}:
            values["disk_io"] = "low"
        values.update(self.extra.get("control", {}))
        return values

    def system(self) -> dict:
        cfg = self.config
        values: dict[str, Any] = {
            "ibrav": 0,
            "nat": len(self.structure),
            "ntyp": len(self.species),
            "ecutwfc": float(cfg.get("dft.ecutwfc")),
            "ecutrho": float(cfg.get("dft.ecutrho")),
        }
        occ = self.occupations or cfg.get("dft.occupations", "smearing")
        values["occupations"] = occ
        if occ == "smearing":
            values["smearing"] = cfg.get("dft.smearing", "mv")
            values["degauss"] = float(cfg.get("dft.degauss", 0.02))
        if self.nbnd:
            values["nbnd"] = int(self.nbnd)

        nspin = int(cfg.get("dft.nspin", 1) or 1)
        if nspin == 2:
            from ezcal.structures import label_element

            values["nspin"] = 2
            mags = dict(cfg.get("dft.starting_magnetization", {}) or {})
            for i, label in enumerate(self.species, start=1):
                # 副格子ラベルの指定は、素の元素名の指定より優先する
                magnetization = mags.get(label, mags.get(label_element(label), 0.3))
                values[f"starting_magnetization({i})"] = float(magnetization)
            if cfg.get("dft.tot_magnetization") is not None:
                values["tot_magnetization"] = float(cfg.get("dft.tot_magnetization"))

        if cfg.get("dft.input_dft"):
            values["input_dft"] = cfg.get("dft.input_dft")
        if cfg.get("dft.vdw_corr"):
            values["vdw_corr"] = cfg.get("dft.vdw_corr")
        if cfg.get("dft.assume_isolated"):
            values["assume_isolated"] = cfg.get("dft.assume_isolated")
        if cfg.get("dft.nosym"):
            values["nosym"] = True
            values["noinv"] = True
        values.update(self.extra.get("system", {}))
        return values

    def electrons(self) -> dict:
        cfg = self.config
        values = {
            "conv_thr": float(cfg.get("dft.conv_thr", 1e-8)),
            "mixing_beta": float(cfg.get("dft.mixing_beta", 0.4)),
            "mixing_mode": cfg.get("dft.mixing_mode", "plain"),
            "electron_maxstep": int(cfg.get("dft.electron_maxstep", 200)),
            "diagonalization": cfg.get("dft.diagonalization", "david"),
        }
        if self.startingpot:
            values["startingpot"] = self.startingpot
        if self.startingwfc:
            values["startingwfc"] = self.startingwfc
        values.update(self.extra.get("electrons", {}))
        return values

    def ions(self) -> dict:
        values = {"ion_dynamics": self.config.get("relax.ion_dynamics", "bfgs")}
        values.update(self.extra.get("ions", {}))
        return values

    def cell(self) -> dict:
        cfg = self.config
        values = {
            "cell_dynamics": cfg.get("relax.cell_dynamics", "bfgs"),
            "press": float(cfg.get("relax.press", 0.0)),
            "press_conv_thr": float(cfg.get("relax.press_conv_thr", 0.5)),
            "cell_dofree": cfg.get("relax.cell_dofree", "all"),
        }
        values.update(self.extra.get("cell", {}))
        return values

    # ------------------------------------------------------------- カード
    def card_species(self) -> str:
        from pymatgen.core.periodic_table import Element

        from ezcal.structures import label_element

        lines = ["ATOMIC_SPECIES"]
        for label in self.species:
            element = label_element(label)
            info = self.pseudos[element]
            filename = getattr(info, "filename", str(info))
            mass = float(Element(element).atomic_mass)
            lines.append(f"  {label:<4s} {mass:10.4f}  {filename}")
        return "\n".join(lines)

    def card_positions(self) -> str:
        from ezcal.structures import site_labels

        lines = ["ATOMIC_POSITIONS crystal"]
        for site, label in zip(self.structure, site_labels(self.structure)):
            x, y, z = site.frac_coords
            lines.append(f"  {label:<4s} {x:18.12f} {y:18.12f} {z:18.12f}")
        return "\n".join(lines)

    def card_cell(self) -> str:
        lines = ["CELL_PARAMETERS angstrom"]
        for row in self.structure.lattice.matrix:
            lines.append("  " + " ".join(f"{v:18.12f}" for v in row))
        return "\n".join(lines)

    def card_kpoints(self) -> str:
        if self.kpoints_explicit is not None:
            pts = np.asarray(self.kpoints_explicit, dtype=float)
            weight = 1.0
            lines = ["K_POINTS crystal", f"  {len(pts)}"]
            for kx, ky, kz in pts:
                lines.append(f"  {kx:14.10f} {ky:14.10f} {kz:14.10f} {weight:10.6f}")
            return "\n".join(lines)
        mesh = self.kmesh or [1, 1, 1]
        offset = list(self.config.get("dft.koffset", [0, 0, 0]) or [0, 0, 0])
        return ("K_POINTS automatic\n  "
                + " ".join(str(int(v)) for v in list(mesh) + offset))

    def card_hubbard(self) -> str:
        hubbard = self.config.get("dft.hubbard_u", {}) or {}
        if not hubbard:
            return ""
        from ezcal.structures import label_element

        projection = self.config.get("dft.hubbard_projection", "ortho-atomic")
        lines = [f"HUBBARD ({projection})"]
        for label in self.species:
            # "Fe1" は自身の指定を拾い、無ければ素の "Fe" の指定を使う
            value = hubbard.get(label, hubbard.get(label_element(label)))
            if value is None:
                continue
            lines.append(f"  U {hubbard_manifold(label)} {float(value):.4f}")
        return "\n".join(lines) if len(lines) > 1 else ""

    # ------------------------------------------------------------ 出力
    def to_string(self) -> str:
        blocks = [
            namelist("control", self.control()),
            namelist("system", self.system()),
            namelist("electrons", self.electrons()),
        ]
        if self.calculation in {"relax", "vc-relax"}:
            blocks.append(namelist("ions", self.ions()))
        if self.calculation == "vc-relax":
            blocks.append(namelist("cell", self.cell()))
        blocks += [self.card_species(), self.card_cell(), self.card_positions(),
                   self.card_kpoints()]
        hubbard = self.card_hubbard()
        if hubbard:
            blocks.append(hubbard)
        return "\n".join(blocks) + "\n"

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_string(), encoding="utf-8")
        return path


# ------------------------------------------------------------------ 後処理
def dos_input(prefix: str, outdir: str, config, fname: str) -> str:
    values: dict[str, Any] = {
        "prefix": prefix,
        "outdir": outdir,
        "fildos": fname,
        "DeltaE": float(config.get("dos.deltae", 0.02)),
    }
    degauss = config.get("dos.degauss")
    if degauss:
        values["degauss"] = float(degauss)
        values["ngauss"] = 0
    if config.get("dos.emin") is not None:
        values["Emin"] = float(config.get("dos.emin"))
    if config.get("dos.emax") is not None:
        values["Emax"] = float(config.get("dos.emax"))
    return namelist("dos", values) + "\n"


def projwfc_input(prefix: str, outdir: str, config, filpdos: str) -> str:
    values: dict[str, Any] = {
        "prefix": prefix,
        "outdir": outdir,
        "filpdos": filpdos,
        "DeltaE": float(config.get("dos.deltae", 0.02)),
        "ngauss": 0,
        "degauss": float(config.get("dos.degauss", 0.01) or 0.01),
    }
    return namelist("projwfc", values) + "\n"


def pp_input(prefix: str, outdir: str, plot_num: int, fileout: str,
             spin_component: int | None = None) -> str:
    """``pp.x`` の入力: FFT グリッド上の量を Gaussian cube として書き出す。"""
    inputpp: dict[str, Any] = {"prefix": prefix, "outdir": outdir, "plot_num": plot_num}
    if spin_component is not None:
        inputpp["spin_component"] = spin_component
    plot = {"iflag": 3, "output_format": 6, "fileout": fileout}
    return namelist("inputpp", inputpp) + "\n" + namelist("plot", plot) + "\n"


def bands_pp_input(prefix: str, outdir: str, filband: str, spin: int | None = None) -> str:
    values: dict[str, Any] = {"prefix": prefix, "outdir": outdir, "filband": filband}
    if spin:
        values["spin_component"] = spin
    return namelist("bands", values) + "\n"
