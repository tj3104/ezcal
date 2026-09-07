"""Quantum ESPRESSO エンジン。"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ezcal.engines.base import CalcResult, Engine, EngineError
from ezcal.engines.qe import inputs as qein
from ezcal.engines.qe import outputs as qeout
from ezcal.pseudo import PseudoManager
from ezcal.scheduler import Command, Stage

_EXE = {"pw": "pw.x", "dos": "dos.x", "projwfc": "projwfc.x",
        "bands": "bands.x", "pp": "pp.x"}


class QEEngine(Engine):
    """``pw.x`` と後処理ツール群を駆動する。"""

    name = "qe"
    supported = ("scf", "relax", "vc-relax", "nscf", "bands", "dos", "pdos", "charge")

    def __init__(self, config, scheduler=None) -> None:
        super().__init__(config, scheduler)
        self.prefix = str(config.get("qe.prefix", "ezcal"))
        self._pseudos: dict[str, Any] | None = None
        self.warnings: list[str] = []

    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    # ------------------------------------------------------------ 準備処理
    def executable(self, key: str) -> str:
        explicit = self.config.get(f"qe.commands.{key}")
        if explicit:
            return str(Path(str(explicit)).expanduser())
        bin_dir = self.config.path("qe.bin_dir")
        if bin_dir:
            candidate = bin_dir / _EXE[key]
            if candidate.is_file():
                return str(candidate)
        return _EXE[key]

    def check(self) -> list[str]:
        problems = []
        pw = self.executable("pw")
        if shutil.which(pw) is None and not Path(pw).is_file():
            problems.append(
                f"pw.x が {pw!r} に見つかりません。qe_config.yaml の qe.bin_dir か qe.commands.pw を設定してください"
            )
        return problems

    def pseudo_manager(self) -> PseudoManager:
        cfg = self.config
        # 探索するのはユーザーが明示したディレクトリのみ。Quantum ESPRESSO 付属の
        # pseudo/ フォルダは整備されたライブラリではなくテストデータであり、そこから
        # ファイルを拾うと PSLibrary の選択を黙って上書きしてしまう
        extra = [Path(str(d)).expanduser()
                 for d in (cfg.get("qe.pseudo_extra_dirs") or [])]
        return PseudoManager(
            pseudo_dir=cfg.get("qe.pseudo_dir", "~/.ezcal/pseudo"),
            functional=str(cfg.get("dft.functional", "pbe")),
            preference=cfg.get("qe.pseudo_preference", ["kjpaw", "rrkjus"]),
            download=bool(cfg.get("qe.pseudo_download", True)),
            source=str(cfg.get("qe.pseudo_source", "")) or
            "https://pseudopotentials.quantum-espresso.org/upf_files",
            extra_dirs=extra,
            pseudo_map=cfg.get("qe.pseudo_map", {}) or {},
        )

    def prepare(self, structure) -> dict:
        """擬ポテンシャルを決定し、``auto`` のままのカットオフを埋める。"""
        if self._pseudos is None:
            manager = self.pseudo_manager()
            elements = [str(el) for el in structure.composition.elements]
            self._pseudos = manager.resolve_all(elements)
            safety = float(self.config.get("dft.ecut_safety", 1.0) or 1.0)
            wfc, rho = manager.suggest_cutoffs(self._pseudos.values(), safety=safety)
            dual = max(p.dual for p in self._pseudos.values())

            chosen = self.config.get("dft.ecutwfc")
            if chosen is None:
                self.config.set("dft.ecutwfc", wfc)
                chosen = wfc
            else:
                chosen = float(chosen)
                if chosen < wfc:
                    self._warn(
                        f"ecutwfc {chosen:g} Ry は擬ポテンシャルが要求する {wfc:g} Ry を"
                        "下回っています。PAW/USPP の計算は 'charge is wrong' で失敗したり、"
                        "cdiaghg で破綻したりすることがあります")

            if self.config.get("dft.ecutrho") is None:
                # dual は実際に使うカットオフと整合させる。下げた ecutwfc に擬ポテンシャル
                # 本来の ecutrho を組み合わせると比が不自然になり、補強電荷が発散する
                suggested = rho if chosen >= wfc else chosen * dual
                self.config.set("dft.ecutrho", round(max(suggested, chosen * dual), 1))
        return self._pseudos

    def pseudo_dir(self) -> Path:
        manager_dir = self.config.path("qe.pseudo_dir") or Path("~/.ezcal/pseudo").expanduser()
        return manager_dir

    # ------------------------------------------------------------ バンド数
    def nelec(self, structure) -> float:
        pseudos = self.prepare(structure)
        return float(sum(pseudos[site.specie.symbol].z_valence for site in structure))

    def suggest_nbnd(self, structure, extra_ratio: float = 0.3, minimum_extra: int = 4) -> int:
        nelec = self.nelec(structure)
        occupied = max(1, math.ceil(nelec / 2.0))
        return int(occupied + max(minimum_extra, math.ceil(occupied * extra_ratio)))

    # ------------------------------------------------------------- 実行部
    def run(self, structure, task: str, workdir: Path, prev: CalcResult | None = None,
            **kwargs) -> CalcResult:
        task = task.lower()
        if not self.supports(task):
            raise EngineError(f"QE エンジンはタスク {task!r} を実行できません")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        outdir = Path(kwargs.pop("outdir", None) or workdir / "tmp").resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        if task == "charge":
            return self._run_charge(structure, workdir, outdir, prev, **kwargs)
        if task in {"dos", "pdos"}:
            return self._run_postproc(task, structure, workdir, outdir, prev, **kwargs)
        return self._run_pw(task, structure, workdir, outdir, prev, **kwargs)

    # -- pw.x -------------------------------------------------------------
    def _run_pw(self, task, structure, workdir: Path, outdir: Path,
                prev: CalcResult | None, **kwargs) -> CalcResult:
        cfg = self.config
        pseudos = self.prepare(structure)

        kpath = kwargs.get("kpath")
        kmesh = kwargs.get("kmesh")
        explicit = None
        if task == "bands":
            if kpath is None:
                from ezcal.structures import band_path

                kpath = band_path(
                    structure,
                    line_density=float(cfg.get("bands.line_density", 25)),
                    symprec=float(cfg.get("bands.symprec", 1e-5)),
                    min_points=int(cfg.get("bands.min_points_per_segment", 6)),
                )
            explicit = kpath.kpoints
        elif kmesh is None:
            kmesh = self._kmesh(cfg, structure)

        nbnd = kwargs.get("nbnd")
        if nbnd is None and task in {"nscf", "bands"}:
            ratio = float(cfg.get(f"{'bands' if task == 'bands' else 'nscf'}.nbnd_extra_ratio", 0.3))
            nbnd = self.suggest_nbnd(structure, ratio)
        if cfg.get("dft.nbnd"):
            nbnd = int(cfg.get("dft.nbnd"))

        occupations = kwargs.get("occupations")
        extra: dict[str, dict[str, Any]] = {}
        if task in {"nscf", "bands"}:
            extra.setdefault("electrons", {})["conv_thr"] = (
                float(cfg.get("dft.conv_thr", 1e-8))
                * float(cfg.get("nscf.conv_thr_scale", 0.1))
            )
        if task == "bands":
            occupations = occupations or cfg.get("dft.occupations", "smearing")

        builder = qein.PwInput(
            structure=structure,
            pseudos=pseudos,
            config=cfg,
            calculation=task,
            prefix=self.prefix,
            outdir=str(outdir),
            pseudo_dir=str(self.pseudo_dir()),
            kmesh=kmesh,
            kpoints_explicit=explicit,
            nbnd=nbnd,
            occupations=occupations,
            extra=extra,
            startingpot="file" if task in {"nscf", "bands"} else None,
        )
        infile = workdir / f"{task}.in"
        outfile = workdir / f"{task}.out"
        builder.write(infile)

        stage = Stage(
            name=task,
            workdir=workdir,
            commands=[Command(self.executable("pw"), ["-in", infile.name],
                              stdout=outfile, label=f"pw.x {task}")],
        )
        job = self.scheduler.execute(stage)
        result = CalcResult(task=task, engine=self.name, workdir=workdir,
                            prefix=self.prefix, job_id=job.job_id,
                            submitted_only=job.submitted_only,
                            walltime=job.elapsed)
        result.files["input"] = str(infile)
        result.files["output"] = str(outfile)
        result.messages += [f"warning: {w}" for w in self.warnings]
        self.warnings.clear()
        if job.script:
            result.files["job_script"] = str(job.script)
        if job.submitted_only:
            result.ok = True
            result.messages.append(
                f"ジョブ {job.job_id} としてキューに投入しました" if job.job_id
                else "ドライラン: 入力を書き出しました (pw.x は起動していません)")
            return result
        if not job.ok:
            result.messages += job.log[-2:]
            result.messages.append(f"pw.x が失敗しました (終了コード {job.returncode})。{outfile} を確認してください")
            self._attach_text(result, outfile)
            return result

        self._attach_text(result, outfile)
        xml = outdir / f"{self.prefix}.xml"
        if not xml.is_file():
            xml = outdir / f"{self.prefix}.save" / "data-file-schema.xml"
        if xml.is_file():
            self._attach_xml(result, xml, structure, kmesh)
        result.files["xml"] = str(xml)
        if task == "bands" and kpath is not None:
            result.data["kpath"] = {
                "distances": kpath.distances.tolist(),
                "labels": [[int(i), lab] for i, lab in kpath.labels],
                "kpoints": kpath.kpoints.tolist(),
            }
        result.ok = bool(result.converged or task in {"nscf", "bands"})
        if not result.ok and result.energy is not None:
            result.ok = True
            result.messages.append("計算は終了しましたが、SCF 収束フラグが立っていません")
        return result

    # -- dos.x / projwfc.x -------------------------------------------------
    def _run_postproc(self, task, structure, workdir: Path, outdir: Path,
                      prev: CalcResult | None, **kwargs) -> CalcResult:
        cfg = self.config
        commands: list[Command] = []
        dos_file = workdir / f"{self.prefix}.dos"

        dos_in = workdir / "dos.in"
        dos_in.write_text(qein.dos_input(self.prefix, str(outdir), cfg, str(dos_file)),
                          encoding="utf-8")
        commands.append(Command(self.executable("dos"), ["-in", dos_in.name],
                                stdout=workdir / "dos.out", label="dos.x"))

        want_pdos = task == "pdos" or bool(cfg.get("dos.pdos", True))
        if want_pdos:
            proj_in = workdir / "projwfc.in"
            proj_in.write_text(
                qein.projwfc_input(self.prefix, str(outdir), cfg, str(workdir / "pdos")),
                encoding="utf-8")
            commands.append(Command(self.executable("projwfc"), ["-in", proj_in.name],
                                    stdout=workdir / "projwfc.out", label="projwfc.x"))

        stage = Stage(name=task, workdir=workdir, commands=commands)
        job = self.scheduler.execute(stage)
        result = CalcResult(task=task, engine=self.name, workdir=workdir,
                            prefix=self.prefix, job_id=job.job_id,
                            submitted_only=job.submitted_only, walltime=job.elapsed)
        if job.submitted_only:
            result.ok = True
            return result
        if not job.ok:
            result.messages += job.log[-2:]
            return result

        if dos_file.is_file():
            data = qeout.read_dos(dos_file)
            result.data["dos"] = data
            result.fermi_energy = data.get("fermi_energy")
            result.files["dos"] = str(dos_file)
        if want_pdos:
            pdos = qeout.read_pdos(workdir, "pdos")
            if pdos:
                result.data["pdos"] = pdos
        result.ok = "dos" in result.data
        if not result.ok:
            result.messages.append(f"dos.x が出力を生成しませんでした。{workdir/'dos.out'} を確認してください")
        return result

    # -- 電荷密度と原子電荷 ------------------------------------------------
    def _run_charge(self, structure, workdir: Path, outdir: Path,
                    prev: CalcResult | None, **kwargs) -> CalcResult:
        """pp.x で cube を出力し、Bader ベイスンと projwfc 由来の Loewdin 電荷をまとめる。"""
        from ezcal import charge as chargemod

        cfg = self.config
        result = CalcResult(task="charge", engine=self.name, workdir=workdir,
                            prefix=self.prefix)

        kinds = list(kwargs.get("kinds") or cfg.get("charge.kinds", ["density"]))
        if int(cfg.get("dft.nspin", 1) or 1) == 2 and "spin" not in kinds:
            kinds.append("spin")
        unknown = [k for k in kinds if k not in chargemod.PLOT_NUM]
        if unknown:
            raise EngineError(
                f"unknown charge quantity {unknown[0]!r}; "
                f"choose from {', '.join(chargemod.PLOT_NUM)}")

        commands = []
        cubes: dict[str, Path] = {}
        for kind in kinds:
            cube = workdir / f"{kind}.cube"
            infile = workdir / f"pp_{kind}.in"
            infile.write_text(
                qein.pp_input(self.prefix, str(outdir), chargemod.PLOT_NUM[kind], str(cube)),
                encoding="utf-8")
            commands.append(Command(self.executable("pp"), ["-in", infile.name],
                                    stdout=workdir / f"pp_{kind}.out", label=f"pp.x {kind}"))
            cubes[kind] = cube

        job = self.scheduler.execute(Stage(name="charge", workdir=workdir,
                                           commands=commands))
        result.job_id, result.submitted_only = job.job_id, job.submitted_only
        result.walltime = job.elapsed
        if job.submitted_only:
            result.ok = True
            return result
        if not job.ok:
            result.messages += job.log[-2:]
            result.messages.append(f"pp.x failed (exit code {job.returncode}); see {workdir}")
            return result

        for kind, path in cubes.items():
            if path.is_file():
                result.files[f"cube_{kind}"] = str(path)
        if not result.files:
            result.messages.append("pp.x produced no cube files")
            return result

        self._analyse_charge(result, structure, workdir, cubes, prev)
        result.ok = True
        return result

    def _analyse_charge(self, result: CalcResult, structure, workdir: Path,
                        cubes: dict[str, Path], prev: CalcResult | None) -> None:
        from ezcal import charge as chargemod

        cfg = self.config
        pseudos = self.prepare(structure)
        valence = [float(pseudos[site.specie.symbol].z_valence) for site in structure]
        nelec = float(sum(valence))

        loaded: dict[str, Any] = {}
        for kind, path in cubes.items():
            if not path.is_file():
                continue
            try:
                loaded[kind] = chargemod.read_cube(path)
            except Exception as exc:
                result.messages.append(f"{path.name} を読めませんでした: {exc}")
        result.data["cubes"] = {k: {"grid": c.shape, "electrons": c.electrons}
                                for k, c in loaded.items()}
        density = loaded.get("density")
        if density is not None:
            result.data["nelec_from_grid"] = density.electrons

        # Loewdin 電荷は DOS ステップの projwfc 出力に既に含まれている
        lowdin = {}
        for candidate in self._projwfc_candidates(workdir, prev):
            lowdin = chargemod.parse_lowdin(candidate)
            if lowdin:
                result.files["projwfc"] = str(candidate)
                break
        if lowdin:
            result.data["lowdin"] = lowdin
        else:
            result.messages.append(
                "projwfc.x の出力が見つからないため Loewdin 電荷は得られません。先に dos ステップを実行してください")

        bader = None
        source = str(cfg.get("charge.bader_source", "density"))
        if cfg.get("charge.bader", True) and source in loaded:
            reference = valence if source in {"density", "ae_valence"} else [
                float(site.specie.Z) for site in structure]
            try:
                bader = chargemod.bader_charges(
                    loaded[source], reference=reference,
                    expected_electrons=nelec if source in {"density", "ae_valence"} else None)
            except Exception as exc:
                result.messages.append(f"Bader 解析に失敗しました: {exc}")
            else:
                result.data["bader"] = {
                    "source": source, "grid": bader.grid,
                    "n_maxima": bader.n_maxima, "n_maxima_raw": bader.n_maxima_raw,
                    "max_offset": bader.max_offset,
                    "total_electrons": bader.total_electrons,
                    "electrons": bader.electrons.tolist(),
                    "volumes": bader.volumes.tolist(),
                    "charges": None if bader.charges is None else bader.charges.tolist(),
                }
                result.messages += bader.messages

        moments = (prev.site_magnetization if prev is not None else None) or None
        rows = chargemod.atomic_charge_table(structure, lowdin, bader, moments, valence)
        result.data["atoms"] = rows
        result.files["charges_csv"] = str(
            chargemod.write_charge_csv(rows, workdir / "atomic_charges.csv"))

    @staticmethod
    def _projwfc_candidates(workdir: Path, prev: CalcResult | None):
        """projwfc.x の出力があり得る場所: 今回のステップ、直前のステップ、同階層。"""
        seen: list[Path] = []
        roots = [workdir]
        if prev is not None:
            roots.append(Path(prev.workdir))
            roots += sorted(Path(prev.workdir).parent.glob("*_dos"))
        roots += sorted(workdir.parent.glob("*_dos"))
        for root in roots:
            candidate = Path(root) / "projwfc.out"
            if candidate.is_file() and candidate not in seen:
                seen.append(candidate)
        return seen

    # -- 補助関数 ----------------------------------------------------------
    def _attach_text(self, result: CalcResult, outfile: Path) -> None:
        text = qeout.parse_pw_text(outfile)
        result.converged = text.converged
        if text.walltime:
            result.walltime = text.walltime
        if text.site_magnetization:
            result.site_magnetization = text.site_magnetization
        if text.absolute_magnetization is not None:
            result.abs_magnetization = text.absolute_magnetization
        result.messages += [f"error: {e}" for e in text.errors]
        result.messages += [f"warning: {w}" for w in text.warnings]
        result.messages += [f"hint: {h}" for h in text.hints]
        if text.scf_energies:
            result.data["scf_energies"] = text.scf_energies

    def _attach_xml(self, result: CalcResult, xml: Path, structure,
                    kmesh: Sequence[int] | None) -> None:
        parsed = qeout.parse_xml(xml)
        if result.task not in {"nscf", "bands"} and parsed.energy is not None:
            result.energy = parsed.energy
            result.energy_per_atom = parsed.energy / max(1, len(structure))
        result.fermi_energy = parsed.fermi_energy
        result.homo, result.lumo = parsed.homo, parsed.lumo
        result.band_gap = parsed.band_gap
        result.nelec = parsed.nelec
        result.nbnd = parsed.nbnd
        if result.task not in {"nscf", "bands"}:
            # nscf/bands は scf の電荷密度をそのまま使い、磁化を計算し直さない。
            # QE はその場合、磁化としてただ 0 を書き出す
            result.magnetization = parsed.magnetization
            if parsed.absolute_magnetization is not None:
                result.abs_magnetization = parsed.absolute_magnetization
        result.kmesh = list(kmesh) if kmesh else parsed.kmesh
        if parsed.forces is not None:
            result.forces = parsed.forces.tolist()
            result.max_force = float(np.abs(parsed.forces).max())
        if parsed.stress is not None:
            result.stress = parsed.stress.tolist()
            result.pressure = parsed.pressure
        if parsed.eigenvalues is not None:
            result.nkpt = int(parsed.eigenvalues.shape[1])
            result.data["eigenvalues"] = parsed.eigenvalues
            result.data["kpoints_frac"] = parsed.kpoints_frac
            result.data["occupations"] = parsed.occupations
            gap = band_gap_from_eigenvalues(parsed)
            if gap is not None:
                result.band_gap = gap["gap"]
                result.data["gap_info"] = gap
                if gap.get("metal"):
                    result.homo = result.lumo = None
                else:
                    result.homo, result.lumo = gap["vbm"], gap["cbm"]
        relaxed = parsed.structure()
        if relaxed is not None:
            result.structure = relaxed
        if parsed.converged:
            result.converged = True


def band_gap_from_eigenvalues(parsed: qeout.PwXml) -> dict | None:
    """固有値スペクトルと電子数からバンドギャップを求める。

    E_F との比較ではなく電子数を数えることが重要である。スメアリングを使うと
    Quantum ESPRESSO は絶縁体の E_F を価電子帯上端より *下* に置くため、そのまま
    比較するとあらゆる半導体が金属に見えてしまう。
    """
    eig = parsed.eigenvalues
    if eig is None or eig.size == 0 or parsed.nelec is None:
        return None
    nspin, nkpt, nbnd = eig.shape
    nelec = float(parsed.nelec)

    if nspin == 1:
        nocc = nelec / 2.0
        if abs(nocc - round(nocc)) > 1e-6:            # 電子数が奇数 -> 金属
            return {"gap": 0.0, "metal": True, "vbm": None, "cbm": None, "direct": None}
        nocc = int(round(nocc))
        if nocc < 1 or nocc >= nbnd:
            return None
        homo_k = eig[0, :, nocc - 1]
        lumo_k = eig[0, :, nocc]
    else:
        if abs(nelec - round(nelec)) > 1e-6:
            return {"gap": 0.0, "metal": True, "vbm": None, "cbm": None, "direct": None}
        nocc = int(round(nelec))
        pooled = np.sort(eig.transpose(1, 0, 2).reshape(nkpt, -1), axis=1)
        if nocc < 1 or nocc >= pooled.shape[1]:
            return None
        homo_k = pooled[:, nocc - 1]
        lumo_k = pooled[:, nocc]

    vbm, cbm = float(homo_k.max()), float(lumo_k.min())
    k_vbm, k_cbm = int(np.argmax(homo_k)), int(np.argmin(lumo_k))
    if cbm <= vbm:                                     # バンドが重なっている -> 金属
        return {"gap": 0.0, "metal": True, "vbm": vbm, "cbm": cbm, "direct": None}
    return {
        "gap": cbm - vbm,
        "metal": False,
        "vbm": vbm,
        "cbm": cbm,
        "direct": bool(k_vbm == k_cbm),
        "k_vbm": k_vbm,
        "k_cbm": k_cbm,
    }
