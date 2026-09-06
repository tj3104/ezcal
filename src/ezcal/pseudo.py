"""Pseudopotential management.

ezcal ships an index of every UPF file published on
``pseudopotentials.quantum-espresso.org`` (the PSLibrary / legacy tables).
Missing files are downloaded on demand into ``qe.pseudo_dir`` and their
headers are parsed so that plane-wave cutoffs, valence electron counts and
the pseudopotential type are known without the user having to look
anything up.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

INDEX_PATH = Path(__file__).with_name("data") / "upf_index.json"
DEFAULT_SOURCE = "https://pseudopotentials.quantum-espresso.org/upf_files"

#: functional keyword -> tokens accepted in a UPF file name
FUNCTIONAL_ALIASES = {
    "pbe": ("pbe",),
    "pbesol": ("pbesol",),
    "pz": ("pz",),
    "lda": ("pz",),
    "blyp": ("blyp",),
    "bp": ("bp",),
    "revpbe": ("revpbe",),
}

_TYPE_TOKENS = (
    ("kjpaw", "PAW"),
    ("_paw", "PAW"),
    ("rrkjus", "USPP"),
    ("van", "USPP"),
    ("us", "USPP"),
    ("oncv", "NC"),
    ("bhs", "NC"),
    ("gth", "NC"),
    ("mt", "NC"),
    ("nc", "NC"),
)

#: charge-density / wavefunction cutoff ratio per pseudopotential type
DUAL = {"PAW": 8.0, "USPP": 8.0, "NC": 4.0}


class PseudoError(RuntimeError):
    pass


@dataclass
class PseudoInfo:
    """Everything ezcal needs to know about one UPF file."""

    element: str
    filename: str
    path: Path
    pseudo_type: str = "USPP"
    z_valence: float = 0.0
    functional: str = ""
    ecutwfc: float | None = None      # Ry, as suggested by the generator
    ecutrho: float | None = None

    @property
    def dual(self) -> float:
        return DUAL.get(self.pseudo_type, 8.0)


# --------------------------------------------------------------- index I/O
def load_index() -> dict[str, list[str]]:
    if not INDEX_PATH.is_file():
        return {}
    with INDEX_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _classify(filename: str) -> str:
    low = filename.lower()
    for token, kind in _TYPE_TOKENS:
        if token in low:
            return kind
    return "USPP"


def _psl_version(filename: str) -> tuple[int, ...]:
    match = re.search(r"psl\.([0-9.]+?)\.UPF$", filename, re.IGNORECASE)
    if not match:
        return (0,)
    return tuple(int(p) for p in match.group(1).split(".") if p.isdigit())


def _matches_functional(element: str, filename: str, functional: str,
                        relativistic: bool) -> bool:
    rest = filename[len(element) + 1:].lower() if filename.lower().startswith(
        element.lower() + ".") else filename.lower()
    is_rel = rest.startswith("rel-")
    if is_rel != relativistic:
        return False
    if is_rel:
        rest = rest[4:]
    tokens = FUNCTIONAL_ALIASES.get(functional.lower(), (functional.lower(),))
    return any(rest.startswith(tok + "-") or rest.startswith(tok + ".") for tok in tokens)


def candidates(element: str, functional: str = "pbe", relativistic: bool = False,
               index: Mapping[str, list[str]] | None = None) -> list[str]:
    index = index if index is not None else load_index()
    files = index.get(element, [])
    return [f for f in files if _matches_functional(element, f, functional, relativistic)]


def rank_candidates(element: str, files: Sequence[str],
                    preference: Sequence[str] = ("kjpaw", "rrkjus", "oncv")) -> list[str]:
    """Sort candidate file names, best first."""
    pref = [p.lower() for p in preference]

    def key(name: str):
        low = name.lower()
        pref_rank = next((i for i, p in enumerate(pref) if p in low), len(pref))
        version = _psl_version(name)
        return (pref_rank, -sum(v * 100 ** (2 - i) for i, v in enumerate(version[:3])), name)

    return sorted(files, key=key)


# ------------------------------------------------------------ UPF parsing
_CUT_WFC_RE = re.compile(r"cutoff for wavefunctions?:\s*([0-9.]+)", re.IGNORECASE)
_CUT_RHO_RE = re.compile(r"cutoff for charge density:\s*([0-9.]+)", re.IGNORECASE)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def parse_upf(path: Path, element: str | None = None) -> PseudoInfo:
    """Read the informative part of a UPF header (first ~200 lines)."""
    head_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            head_lines.append(line)
            if i > 400 or "<PP_MESH" in line:
                break
    head = "".join(head_lines)

    info = PseudoInfo(
        element=element or path.name.split(".")[0],
        filename=path.name,
        path=path,
        pseudo_type=_classify(path.name),
    )

    match = _CUT_WFC_RE.search(head)
    if match:
        info.ecutwfc = float(match.group(1))
    match = _CUT_RHO_RE.search(head)
    if match:
        info.ecutrho = float(match.group(1))

    header = re.search(r"<PP_HEADER(.*?)/?>", head, re.DOTALL)
    if header:
        attrs = dict(_ATTR_RE.findall(header.group(1)))
        if "z_valence" in attrs:
            try:
                info.z_valence = float(attrs["z_valence"])
            except ValueError:
                pass
        if attrs.get("pseudo_type"):
            raw = attrs["pseudo_type"].upper()
            info.pseudo_type = {"PAW": "PAW", "US": "USPP", "USPP": "USPP",
                                "NC": "NC", "SL": "NC"}.get(raw, info.pseudo_type)
        info.functional = attrs.get("functional", "").strip()
        if attrs.get("wfc_cutoff") and not info.ecutwfc:
            try:
                info.ecutwfc = float(attrs["wfc_cutoff"])
            except ValueError:
                pass
        if attrs.get("rho_cutoff") and not info.ecutrho:
            try:
                info.ecutrho = float(attrs["rho_cutoff"])
            except ValueError:
                pass

    if not info.z_valence:                       # UPF v1 fixed format fallback
        match = re.search(r"([0-9.]+)\s+Z valence", head)
        if match:
            info.z_valence = float(match.group(1))
    if not info.functional:
        match = re.search(r"Functional:\s*(.+)", head)
        if match:
            info.functional = match.group(1).strip()
    return info


# ----------------------------------------------------------------- manager
class PseudoManager:
    """Resolve element -> UPF file, downloading from the QE website if needed."""

    def __init__(
        self,
        pseudo_dir: str | Path = "~/.ezcal/pseudo",
        functional: str = "pbe",
        preference: Sequence[str] = ("kjpaw", "rrkjus", "oncv"),
        download: bool = True,
        source: str = DEFAULT_SOURCE,
        extra_dirs: Iterable[str | Path] = (),
        pseudo_map: Mapping[str, str] | None = None,
        relativistic: bool = False,
    ) -> None:
        self.pseudo_dir = Path(pseudo_dir).expanduser()
        self.functional = functional
        self.preference = list(preference)
        self.download = download
        self.source = source.rstrip("/")
        self.extra_dirs = [Path(d).expanduser() for d in extra_dirs]
        self.pseudo_map = dict(pseudo_map or {})
        self.relativistic = relativistic
        self.index = load_index()
        self._cache: dict[str, PseudoInfo] = {}

    # -- lookup ---------------------------------------------------------
    def _search_dirs(self) -> list[Path]:
        return [self.pseudo_dir, *self.extra_dirs]

    def _find_local(self, element: str, filename: str | None = None) -> Path | None:
        for directory in self._search_dirs():
            if not directory.is_dir():
                continue
            if filename:
                candidate = directory / filename
                if candidate.is_file():
                    return candidate
                continue
            matches = [
                p for p in directory.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".upf", ".psp8"}
                and p.name.split(".")[0].lower() == element.lower()
                and _matches_functional(element, p.name, self.functional, self.relativistic)
            ]
            if matches:
                ranked = rank_candidates(element, [p.name for p in matches], self.preference)
                return next(p for p in matches if p.name == ranked[0])
        return None

    def _adopt(self, path: Path) -> Path:
        """Copy a file found in an extra directory into ``pseudo_dir``.

        The generated pw.x input names the file and sets ``pseudo_dir`` once,
        so a pseudopotential picked up somewhere else has to be brought in or
        pw.x fails with 'file ... not found'.
        """
        import shutil

        if path.parent.resolve() == self.pseudo_dir.resolve():
            return path
        self.pseudo_dir.mkdir(parents=True, exist_ok=True)
        target = self.pseudo_dir / path.name
        if not target.is_file():
            shutil.copy2(path, target)
        return target

    def _fetch(self, filename: str) -> Path:
        self.pseudo_dir.mkdir(parents=True, exist_ok=True)
        target = self.pseudo_dir / filename
        url = f"{self.source}/{filename}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ezcal"})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PseudoError(f"could not download {url}: {exc}") from exc
        if len(payload) < 1000 or b"<UPF" not in payload[:4000] and b"<PP_INFO" not in payload[:4000]:
            raise PseudoError(f"{url} did not return a UPF file")
        target.write_bytes(payload)
        return target

    def resolve(self, element: str) -> PseudoInfo:
        """Return a :class:`PseudoInfo` for ``element``, downloading if allowed."""
        if element in self._cache:
            return self._cache[element]

        pinned = self.pseudo_map.get(element)
        if pinned:
            pinned_path = Path(pinned).expanduser()
            if pinned_path.is_file():
                info = parse_upf(pinned_path, element)
                self._cache[element] = info
                return info
            local = self._find_local(element, filename=pinned)
            if local is None and self.download:
                local = self._fetch(pinned)
            if local is None:
                raise PseudoError(f"pinned pseudopotential {pinned} not found for {element}")
            info = parse_upf(self._adopt(local), element)
            self._cache[element] = info
            return info

        local = self._find_local(element)
        if local is not None:
            info = parse_upf(self._adopt(local), element)
            self._cache[element] = info
            return info

        options = rank_candidates(
            element,
            candidates(element, self.functional, self.relativistic, self.index),
            self.preference,
        )
        if not options:
            raise PseudoError(
                f"no {self.functional.upper()} pseudopotential known for {element}; "
                f"put a UPF file in {self.pseudo_dir} or pin one with "
                f"--set qe.pseudo_map.{element}=<file.UPF>"
            )
        if not self.download:
            raise PseudoError(
                f"{element}: {options[0]} is missing from {self.pseudo_dir} and "
                "downloading is disabled (qe.pseudo_download: false)"
            )
        path = self._fetch(options[0])
        info = parse_upf(path, element)
        self._cache[element] = info
        return info

    def resolve_all(self, elements: Iterable[str]) -> dict[str, PseudoInfo]:
        return {el: self.resolve(el) for el in dict.fromkeys(elements)}

    # -- derived quantities ---------------------------------------------
    @staticmethod
    def suggest_cutoffs(infos: Iterable[PseudoInfo], safety: float = 1.0,
                        fallback_wfc: float = 60.0) -> tuple[float, float]:
        """Largest generator-suggested cutoff over all species, times ``safety``."""
        infos = list(infos)
        wfc = max((i.ecutwfc for i in infos if i.ecutwfc), default=None) or fallback_wfc
        duals = [i.dual for i in infos] or [8.0]
        rho_from_dual = wfc * max(duals)
        rho = max((i.ecutrho for i in infos if i.ecutrho), default=0.0)
        rho = max(rho, rho_from_dual)
        return round(wfc * safety, 1), round(rho * safety, 1)

    @staticmethod
    def valence_electrons(structure, infos: Mapping[str, PseudoInfo]) -> float:
        total = 0.0
        for site in structure:
            symbol = site.specie.symbol
            info = infos.get(symbol)
            if info is not None:
                total += info.z_valence
        return total
