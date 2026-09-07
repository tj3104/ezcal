"""VASP エンジン。

``vasp.command`` が指すものを実行する。通常はライセンス済みの ``vasp_std``
バイナリだが、aiida-vasp の ``mock-vasp`` でも構わない。後者は代役であり、
作業ディレクトリの INCAR/KPOINTS/POSCAR を解析してハッシュを取り、完了済み計算の
レジストリから該当するものを探して、記録済みの出力を書き戻す。これにより VASP 経路
全体 (入力生成・実行・解析・作図) を、ライセンス無しで検証できるようになり、
研究室内で参照計算を再生することもできる。

出所は隠さない。モック由来の結果には ``data["mock"]`` が付き、実行レポートにも
その旨が明記される。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ezcal.engines.base import CalcResult, Engine, EngineError
from ezcal.scheduler import Command, Stage

RY_EV = 13.605693122994
BOHR_ANG = 0.529177210903
KBAR_GPA = 0.1

#: ezcal のエンジン非依存オプションと INCAR タグの対応 (全タスク共通)
INCAR_MAP: dict[str, tuple[str, Any]] = {
    "dft.ecutwfc": ("ENCUT", lambda ry: round(float(ry) * RY_EV, 1)),        # Ry -> eV
    "dft.conv_thr": ("EDIFF", lambda v: float(v) * RY_EV),                   # Ry -> eV
    "dft.degauss": ("SIGMA", lambda ry: round(float(ry) * RY_EV, 4)),        # Ry -> eV
    "dft.nbnd": ("NBANDS", int),
    "dft.electron_maxstep": ("NELM", int),
}

#: 原子を動かすタスクでのみ対応付けるもの
RELAX_INCAR_MAP: dict[str, tuple[str, Any]] = {
    "relax.nstep": ("NSW", int),
    # Ry/Bohr -> eV/Angstrom。VASP は負値を力の収束条件として解釈するため符号を反転
    "relax.forc_conv_thr": ("EDIFFG", lambda v: -abs(float(v) * RY_EV / BOHR_ANG)),
}

#: ezcal 側のオプションが同梱の既定値 (Quantum ESPRESSO 寄りの値) のままだった
#: 場合に使う、VASP にとって妥当な出発点
VASP_DEFAULTS: dict[str, Any] = {"EDIFF": 1.0e-6}

#: ezcal のタスク -> そのタスクを規定する INCAR 設定
TASK_INCAR: dict[str, dict[str, Any]] = {
    "scf": {"IBRION": -1, "NSW": 0, "ISIF": 2, "LCHARG": True, "LWAVE": True},
    "relax": {"IBRION": 2, "NSW": 100, "ISIF": 2, "LCHARG": True},
    "vc-relax": {"IBRION": 2, "NSW": 100, "ISIF": 3, "LCHARG": True},
    "nscf": {"IBRION": -1, "NSW": 0, "ICHARG": 11, "ISMEAR": -5, "LORBIT": 11},
    "bands": {"IBRION": -1, "NSW": 0, "ICHARG": 11, "ISMEAR": 0, "LORBIT": 11},
    "dos": {"IBRION": -1, "NSW": 0, "ICHARG": 11, "ISMEAR": -5, "LORBIT": 11},
}

#: 占有数の方式 -> ISMEAR
ISMEAR = {"smearing": 1, "gaussian": 0, "mv": 1, "mp": 1, "fd": -1,
          "fixed": 0, "tetrahedra": -5}

_MOCK_REGISTRY_RE = re.compile(r"Using test data in path (\S+) based detection")
_MOCK_CASE_RE = re.compile(r"Using test data from folder: (\S+)")
_MOCK_DEFAULT = "Using default test data"

PLACEHOLDER_POTCAR = """\
これは VASP の擬ポテンシャルではありません。

設定された VASP コマンドがモック ({command}) であり、ファイルの存在確認しか
行わないため、ezcal がこのダミーファイルを書き出しました。実際の VASP 計算には
ライセンス済みの POTCAR ライブラリが必要です。qe_config.yaml で次のように
指定してください。

    vasp:
      potcar_dir: /path/to/potpaw_PBE
      potcar_mode: library

この計算に含まれる元素 (POSCAR の順): {elements}
"""


class VaspEngine(Engine):
    """VASP (または aiida-vasp の ``mock-vasp``) を実行し、``vasprun.xml`` を解析する。"""

    name = "vasp"
    supported = ("scf", "relax", "vc-relax", "nscf", "bands", "dos", "pdos")

    # ------------------------------------------------------------ コマンド
    def command(self) -> str:
        return str(self.config.get("vasp.command", "vasp_std"))

    def is_mock(self) -> bool:
        """設定されたコマンドが aiida-vasp のモック実行ファイルなら True。"""
        return Path(self.command()).name.startswith("mock-vasp")

    def check(self) -> list[str]:
        import shutil

        command = self.command()
        problems: list[str] = []
        if shutil.which(command) is None and not Path(command).is_file():
            problems.append(
                f"VASP コマンド {command!r} が見つかりません。qe_config.yaml の vasp.command を"
                "設定してください (ライセンス無しで VASP 経路を試すなら aiida-vasp の "
                "'mock-vasp' が使えます)"
            )
        if self.is_mock() and not self.config.get("vasp.mock.registry"):
            problems.append(
                "vasp.command がモックですが vasp.mock.registry が未設定のため、再生できる"
                "ものがありません。レジストリのディレクトリを指定してください"
            )
        return problems

    # ---------------------------------------------------------- INCAR
    def incar(self, structure, task: str) -> dict[str, Any]:
        """1 タスク分の INCAR タグを組み立てる。

        ``vasp.incar`` は個々のタグを上書きする (値が ``null`` なら削除)。
        ``vasp.incar_mode: replace`` にすると、それ自体が INCAR 全体になる。
        記録済み計算をバイト単位で再現したい場合はこちらを使う。
        """
        overrides = dict(self.config.get("vasp.incar", {}) or {})
        if str(self.config.get("vasp.incar_mode", "merge")).lower() == "replace":
            return {key: value for key, value in overrides.items() if value is not None}

        cfg = self.config
        tags: dict[str, Any] = {"PREC": "Accurate", "LREAL": "Auto", "ALGO": "Normal"}
        tags.update(VASP_DEFAULTS)

        maps = dict(INCAR_MAP)
        if task in {"relax", "vc-relax"}:
            maps.update(RELAX_INCAR_MAP)
        for dotted, (tag, convert) in maps.items():
            value = cfg.get(dotted)
            # 手つかずの QE 寄りの既定値を VASP に持ち込んではいけない。conv_thr の
            # 1e-8 Ry は EDIFF 1.4e-7 eV に相当し、VASP では非現実的な値になる
            if value is None or (tag in VASP_DEFAULTS and not _is_user_set(dotted, value)):
                continue
            tags[tag] = convert(value)

        # 計算の種類を決めるのはタスクなので、汎用の対応付けより優先させる
        tags.update(TASK_INCAR.get(task, {}))

        if task in {"scf", "relax", "vc-relax"}:
            occupations = str(cfg.get("dft.occupations", "smearing"))
            smearing = str(cfg.get("dft.smearing", "mv"))
            tags["ISMEAR"] = ISMEAR.get(occupations if occupations != "smearing" else smearing, 1)

        if int(cfg.get("dft.nspin", 1) or 1) == 2:
            from ezcal.structures import label_element, site_labels

            tags["ISPIN"] = 2
            magmoms = cfg.get("dft.starting_magnetization", {}) or {}
            # VASP はサイトごとの初期モーメントを、POSCAR の順にボーア磁子単位で要求する。
            # QE は同じオプションを価電子数に対する割合として読むが、ezcal は独自の換算を
            # 作らず値をそのまま渡す。両コードで本質的に重要なのは、磁気秩序を決める符号の
            # パターンだからである。指定のないサイトには VASP 既定の 1.0 が入る。
            default = float(cfg.get("vasp.magmom_default", 1.0) or 1.0)
            per_site = []
            for label in site_labels(structure):
                value = magmoms.get(label, magmoms.get(label_element(label)))
                per_site.append(round(float(default if value is None else value), 4))
            tags["MAGMOM"] = " ".join(f"{v:g}" for v in per_site)

        hubbard = cfg.get("dft.hubbard_u", {}) or {}
        if hubbard:
            from ezcal.structures import label_element

            symbols = _poscar_species(structure)
            tags["LDAU"] = True
            tags["LDAUTYPE"] = 2
            tags["LDAUL"] = " ".join(
                str(_hubbard_l(label_element(s))) if _hubbard_value(hubbard, s) else "-1"
                for s in symbols)
            tags["LDAUU"] = " ".join(f"{_hubbard_value(hubbard, s) or 0.0:g}" for s in symbols)
            tags["LDAUJ"] = " ".join("0" for _ in symbols)
            tags["LMAXMIX"] = 4

        if cfg.get("dft.vdw_corr"):
            mapping = {"grimme-d3": {"IVDW": 12}, "grimme-d2": {"IVDW": 10},
                       "ts": {"IVDW": 20}}
            tags.update(mapping.get(str(cfg.get("dft.vdw_corr")).lower(), {}))

        for key, value in overrides.items():
            if value is None:
                tags.pop(key, None)
            else:
                tags[key] = value
        return tags

    # ---------------------------------------------------------- 入力生成
    def write_inputs(self, structure, task: str, workdir: Path,
                     kmesh: Sequence[int] | None = None, kpath=None) -> Path:
        from pymatgen.io.vasp.inputs import Poscar

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        Poscar(structure).write_file(str(workdir / "POSCAR"))
        # INCAR と KPOINTS は pymatgen 経由ではなくここで書き出す。整数を整数のまま保ち、
        # シフト行を必ず含めるため。mock-vasp は *解析後* の入力をハッシュ化するので、
        # 200 と 200.0 は別物として扱われてしまう
        (workdir / "INCAR").write_text(render_incar(self.incar(structure, task)),
                                       encoding="utf-8")
        if kpath is not None:
            text = render_kpoints_explicit(np.asarray(kpath.kpoints, dtype=float))
        else:
            mesh = [int(v) for v in (kmesh or self._kmesh(self.config, structure))]
            shift = [float(v) for v in (self.config.get("dft.koffset") or [0, 0, 0])]
            style = str(self.config.get("vasp.kpoints_style", "gamma")).lower()
            text = render_kpoints_mesh(mesh, shift,
                                       "Monkhorst" if style.startswith("m") else "Gamma")
        (workdir / "KPOINTS").write_text(text, encoding="utf-8")

        self.write_potcar(structure, workdir)
        return workdir

    def write_potcar(self, structure, workdir: Path) -> Path:
        """ライセンス済みライブラリから POTCAR を書き出す。無ければ明示的なダミーを置く。"""
        symbols = _poscar_species(structure)
        mode = str(self.config.get("vasp.potcar_mode", "auto")).lower()
        potcar_dir = self.config.path("vasp.potcar_dir")
        target = Path(workdir) / "POTCAR"

        if mode == "auto":
            mode = "library" if potcar_dir else ("placeholder" if self.is_mock() else "missing")

        if mode == "library":
            if not potcar_dir:
                raise EngineError("vasp.potcar_mode が 'library' ですが vasp.potcar_dir が未設定です")
            from pymatgen.io.vasp.inputs import Potcar

            mapping = self.config.get("vasp.potcar_map", {}) or {}
            names = [str(mapping.get(s, s)) for s in symbols]
            os.environ.setdefault("PMG_VASP_PSP_DIR", str(potcar_dir))
            potcar = Potcar(symbols=names,
                            functional=str(self.config.get("vasp.potcar_functional", "PBE")))
            potcar.write_file(str(target))
            return target

        if mode == "placeholder":
            target.write_text(
                PLACEHOLDER_POTCAR.format(command=self.command(), elements=", ".join(symbols)),
                encoding="utf-8")
            return target

        raise EngineError(
            "利用できる POTCAR がありません。vasp.potcar_dir にライセンス済みライブラリを"
            "指定するか、POTCAR を必要としないモックコマンド (vasp.command: mock-vasp) を"
            "使ってください"
        )

    # ---------------------------------------------------------- 実行
    def registry_path(self) -> Path | None:
        """モックレジストリのディレクトリ。無ければ None。

        mock-vasp 5.1 は単一のベースパスしか受け付けない (``MOCK_VASP_REG_BASE`` の
        中身に対して ``Path()`` を呼ぶ) ため、設定がリストの場合は先頭要素を使う。
        """
        registry = self.config.get("vasp.mock.registry")
        if not registry:
            return None
        if isinstance(registry, (list, tuple)):
            registry = registry[0]
        return Path(os.path.expandvars(str(registry))).expanduser().resolve()

    def stage_env(self) -> dict[str, str]:
        """モック用の環境変数。実 VASP の場合は空。"""
        env: dict[str, str] = {}
        registry = self.registry_path()
        if registry:
            env["MOCK_VASP_REG_BASE"] = str(registry)
        vasp_cmd = self.config.get("vasp.mock.vasp_cmd")
        if vasp_cmd:
            env["MOCK_VASP_VASP_CMD"] = str(vasp_cmd)
        return env

    def run(self, structure, task: str, workdir: Path, prev: CalcResult | None = None,
            **kwargs) -> CalcResult:
        task = task.lower()
        if not self.supports(task):
            raise EngineError(f"VASP エンジンはタスク {task!r} を実行できません")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        kpath = kwargs.get("kpath")
        if task == "bands" and kpath is None:
            from ezcal.structures import band_path

            kpath = band_path(
                structure,
                line_density=float(self.config.get("bands.line_density", 25)),
                symprec=float(self.config.get("bands.symprec", 1e-5)),
                min_points=int(self.config.get("bands.min_points_per_segment", 6)),
            )
        self.write_inputs(structure, task, workdir, kmesh=kwargs.get("kmesh"), kpath=kpath)

        # bands/dos の計算は直前の scf の電荷密度を読む
        if task in {"nscf", "bands", "dos"} and prev is not None:
            _link_chgcar(Path(prev.workdir), workdir)

        stage = Stage(
            name=task,
            workdir=workdir,
            env=self.stage_env(),
            commands=[Command(self.command(), [], stdout=workdir / "vasp.out",
                              label=f"vasp {task}")],
        )
        job = self.scheduler.execute(stage)

        result = CalcResult(task=task, engine=self.name, workdir=workdir,
                            job_id=job.job_id, submitted_only=job.submitted_only,
                            walltime=job.elapsed)
        for name in ("INCAR", "POSCAR", "KPOINTS"):
            result.files[name.lower()] = str(workdir / name)
        result.files["output"] = str(workdir / "vasp.out")
        if job.script:
            result.files["job_script"] = str(job.script)

        if job.submitted_only:
            result.ok = True
            result.messages.append(
                f"ジョブ {job.job_id} として投入しました" if job.job_id
                else "ドライラン: VASP の入力を書き出しました (実行はしていません)")
            return result
        if not job.ok:
            result.messages += job.log[-2:]
            result.messages.append(f"VASP が失敗しました (終了コード {job.returncode})。{workdir} を確認してください")
            return result

        self._note_mock(result, workdir)
        self._parse(result, workdir, structure, kpath)
        return result

    # ---------------------------------------------------------- 出所の記録
    def _note_mock(self, result: CalcResult, workdir: Path) -> None:
        if not self.is_mock():
            return
        log = workdir / "vasp_output"
        text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
        info: dict[str, Any] = {"used": True, "command": self.command()}

        match = _MOCK_REGISTRY_RE.search(text)
        case = _MOCK_CASE_RE.search(text)
        if match:
            info.update(mode="registry", source=match.group(1))
            result.messages.append(
                f"mock-vasp が {match.group(1)} に記録された VASP 出力を再生しました "
                "(入力のハッシュがそのエントリと一致)")
        elif case:
            info.update(mode="test-case", source=case.group(1))
            result.messages.append(f"mock-vasp が同梱のテストケース {case.group(1)!r} を使用しました")
        elif _MOCK_DEFAULT in text:
            info.update(mode="default", source=None)
            result.messages.append(
                "警告: mock-vasp が内蔵のデモデータにフォールバックしました。これは"
                "別の系のデータであり、指定した構造の計算結果ではありません")
        else:
            info.update(mode="unknown", source=None)
            result.messages.append("mock-vasp が使われましたが、出所を特定できませんでした")
        result.data["mock"] = info

    # ---------------------------------------------------------- 出力の解析
    def _parse(self, result: CalcResult, workdir: Path, structure, kpath) -> None:
        from pymatgen.io.vasp.outputs import Vasprun

        xml = workdir / "vasprun.xml"
        if not xml.is_file():
            result.messages.append(f"{workdir} に vasprun.xml がありません")
            return
        try:
            # DOS のブロックには efermi も含まれるので、常に読む価値がある
            run = Vasprun(str(xml), parse_potcar_file=False, parse_dos=True, parse_eigen=True)
        except Exception as exc:                       # 壊れている、または途中で切れた出力
            result.messages.append(f"vasprun.xml を解析できませんでした: {exc}")
            return

        result.files["vasprun"] = str(xml)
        result.converged = bool(run.converged)
        result.ok = True

        if result.task not in {"nscf", "bands"}:
            result.energy = float(run.final_energy)
            result.energy_per_atom = result.energy / max(1, len(run.final_structure))
        result.fermi_energy = float(run.efermi) if run.efermi is not None else None
        result.structure = run.final_structure
        result.nbnd = int(run.parameters.get("NBANDS", 0)) or None
        result.nelec = float(run.parameters.get("NELECT", 0)) or None

        steps = run.ionic_steps or []
        if steps:
            last = steps[-1]
            if last.get("forces") is not None:
                forces = np.asarray(last["forces"], dtype=float)
                result.forces = forces.tolist()
                result.max_force = float(np.abs(forces).max())
            if last.get("stress") is not None:
                stress = np.asarray(last["stress"], dtype=float) * KBAR_GPA   # kBar -> GPa
                result.stress = stress.tolist()
                result.pressure = float(np.trace(stress) / 3.0)
            energies = [s["e_wo_entrp"] for s in steps if s.get("e_wo_entrp") is not None]
            if len(energies) > 1:
                result.data["scf_energies"] = [float(e) for e in energies]

        self._parse_magnetism(result, workdir, run)
        self._parse_eigenvalues(result, run, kpath)
        # QE と違い、VASP は DOS を持つ計算なら必ず vasprun.xml に DOS を書き出す
        self._parse_dos(result, run)

    def _parse_magnetism(self, result: CalcResult, workdir: Path, run) -> None:
        if int(run.parameters.get("ISPIN", 1)) != 2:
            return
        outcar = workdir / "OUTCAR"
        if not outcar.is_file():
            return
        from pymatgen.io.vasp.outputs import Outcar

        try:
            parsed = Outcar(str(outcar))
        except Exception:
            return
        if parsed.total_magnetization is not None:
            result.magnetization = float(parsed.total_magnetization)
        moments = [float(entry.get("tot", 0.0)) for entry in (parsed.magnetization or [])]
        if moments:
            result.site_magnetization = moments
            result.abs_magnetization = float(np.abs(moments).sum())

    def _parse_eigenvalues(self, result: CalcResult, run, kpath) -> None:
        from pymatgen.electronic_structure.core import Spin

        if not run.eigenvalues:
            return
        spins = [Spin.up] + ([Spin.down] if Spin.down in run.eigenvalues else [])
        eig = np.stack([np.asarray(run.eigenvalues[s])[:, :, 0] for s in spins])  # (ns, nk, nb)
        occ = np.stack([np.asarray(run.eigenvalues[s])[:, :, 1] for s in spins])
        result.data["eigenvalues"] = eig
        result.data["occupations"] = occ
        result.data["kpoints_frac"] = np.asarray(run.actual_kpoints, dtype=float)
        result.nkpt = int(eig.shape[1])

        gap = _band_gap(eig, occ, result.fermi_energy)
        if gap is not None:
            result.band_gap = gap["gap"]
            result.data["gap_info"] = gap
            if not gap["metal"]:
                result.homo, result.lumo = gap["vbm"], gap["cbm"]

        if result.task == "bands" and kpath is not None:
            result.data["kpath"] = {
                "distances": kpath.distances.tolist(),
                "labels": [[int(i), label] for i, label in kpath.labels],
                "kpoints": kpath.kpoints.tolist(),
            }

    def _parse_dos(self, result: CalcResult, run) -> None:
        from pymatgen.electronic_structure.core import Orbital, Spin

        try:
            dos = run.complete_dos
        except Exception:
            dos = getattr(run, "tdos", None)
        if dos is None:
            return

        energies = np.asarray(dos.energies, dtype=float)
        densities = dos.densities
        payload: dict[str, Any] = {"energy": energies,
                                   "fermi_energy": float(dos.efermi) if dos.efermi else None}
        if Spin.down in densities:
            payload["dos_up"] = np.asarray(densities[Spin.up], dtype=float)
            payload["dos_down"] = np.asarray(densities[Spin.down], dtype=float)
            payload["dos"] = payload["dos_up"] + payload["dos_down"]
        else:
            payload["dos"] = np.asarray(densities[Spin.up], dtype=float)
        result.data["dos"] = payload

        if not hasattr(dos, "get_element_spd_dos"):
            return
        per_element: dict[str, np.ndarray] = {}
        per_orbital: dict[str, np.ndarray] = {}
        try:
            for element, element_dos in dos.get_element_dos().items():
                per_element[str(element)] = _sum_spins(element_dos.densities)
            for element in dos.structure.composition.elements:
                for orbital, orbital_dos in dos.get_element_spd_dos(element).items():
                    key = f"{element}-{str(orbital).lower()}"
                    per_orbital[key] = _sum_spins(orbital_dos.densities)
        except Exception:
            pass
        if per_element or per_orbital:
            result.data["pdos"] = {"energy": energies, "spin_polarised": Spin.down in densities,
                                   "per_element": per_element, "per_orbital": per_orbital,
                                   "total": payload["dos"]}

    # ------------------------------------------------------------ レジストリ
    def record(self, rundir: Path, name: str) -> Path:
        """完了済みの VASP 計算をモックレジストリに追加し、再生できるようにする。"""
        from aiida_vasp.utils.mock_code import VaspMockRegistry

        base = self.registry_path()
        if base is None:
            raise EngineError("vasp.mock.registry が設定されていません")
        base.mkdir(parents=True, exist_ok=True)

        rundir = Path(rundir)
        missing = [f for f in ("INCAR", "POSCAR", "KPOINTS", "vasprun.xml")
                   if not (rundir / f).is_file()]
        if missing:
            raise EngineError(f"{rundir} は完了した VASP 計算ではありません (不足: {', '.join(missing)})")

        reg = VaspMockRegistry(str(base))
        reg.upload_calc(rundir, name)
        return base / name


def _is_user_set(dotted: str, value: Any) -> bool:
    """``value`` が default_config.yaml の同梱値と異なる場合に True。"""
    from ezcal.config import default_config

    try:
        shipped = default_config().get(dotted)
    except Exception:
        return True
    return shipped != value


# ---------------------------------------------------------------- 入力の書式化
def incar_value(value: Any) -> str:
    """INCAR の値を 1 つ書式化する。整数は整数のまま、真偽値は Fortran 流に出力する。"""
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(incar_value(v) for v in value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.10g}"
        return text if any(c in text for c in ".eE") else text + ".0"
    return str(value)


def render_incar(tags: dict[str, Any]) -> str:
    return "".join(f"{key} = {incar_value(value)}\n"
                   for key, value in sorted(tags.items()) if value is not None)


def render_kpoints_mesh(mesh: Sequence[int], shift: Sequence[float] = (0, 0, 0),
                        centering: str = "Gamma") -> str:
    return (
        "Automatic mesh written by ezcal\n"
        "0\n"
        f"{centering}\n"
        "  " + "  ".join(f"{int(v):d}" for v in mesh) + "\n"
        "  " + "  ".join(f"{float(v):.9f}" for v in shift) + "\n"
    )


def render_kpoints_explicit(points: np.ndarray, weight: float = 1.0) -> str:
    lines = ["ezcal band path", f"{len(points)}", "Reciprocal"]
    for kx, ky, kz in points:
        lines.append(f"  {kx:14.10f} {ky:14.10f} {kz:14.10f} {weight:8.4f}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ 補助関数
def _poscar_species(structure) -> list[str]:
    """POSCAR の順に並べた元素記号。VASP のグループ化に合わせて重複を除く。"""
    from pymatgen.io.vasp.inputs import Poscar

    return [str(s) for s in Poscar(structure).site_symbols]


def _hubbard_value(hubbard, symbol):
    from ezcal.structures import label_element

    value = hubbard.get(symbol, hubbard.get(label_element(symbol)))
    return float(value) if value is not None else None


def _hubbard_l(element: str) -> int:
    from pymatgen.core.periodic_table import Element

    return {"s": 0, "p": 1, "d": 2, "f": 3}.get(Element(element).block, 2)


def _sum_spins(densities) -> np.ndarray:
    return np.asarray(sum(np.asarray(v, dtype=float) for v in densities.values()))


def _link_chgcar(source: Path, target: Path) -> None:
    """直前のステップの CHGCAR/WAVECAR を今回のステップへ引き継ぐ。"""
    import shutil

    for name in ("CHGCAR", "WAVECAR"):
        origin = Path(source) / name
        if origin.is_file() and not (Path(target) / name).exists():
            try:
                os.link(origin, Path(target) / name)
            except OSError:
                shutil.copy2(origin, Path(target) / name)


def _band_gap(eig: np.ndarray, occ: np.ndarray, fermi: float | None,
              tol: float = 1e-3) -> dict | None:
    """QE エンジンと同じ電子数の数え方を、VASP の占有数に対して適用する。"""
    nspin, nkpt, nbnd = eig.shape
    filled = occ > 0.5
    if not filled.any() or filled.all():
        return None
    counts = {int(filled[s, k].sum()) for s in range(nspin) for k in range(nkpt)}
    if len(counts) != 1:                                # バンドが E_F を横切っている
        return {"gap": 0.0, "metal": True, "vbm": None, "cbm": None, "direct": None}
    nocc = counts.pop()
    if nocc < 1 or nocc >= nbnd:
        return None
    homo_k = eig[:, :, nocc - 1].max(axis=0)
    lumo_k = eig[:, :, nocc].min(axis=0)
    vbm, cbm = float(homo_k.max()), float(lumo_k.min())
    if cbm - vbm <= tol:
        return {"gap": 0.0, "metal": True, "vbm": vbm, "cbm": cbm, "direct": None}
    return {"gap": cbm - vbm, "metal": False, "vbm": vbm, "cbm": cbm,
            "direct": bool(int(np.argmax(homo_k)) == int(np.argmin(lumo_k))),
            "k_vbm": int(np.argmax(homo_k)), "k_cbm": int(np.argmin(lumo_k))}
