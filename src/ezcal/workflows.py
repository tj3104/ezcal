"""タスクの連鎖 (チェーン)。

``ezcal bands Si.cif`` は構造ファイルだけで動く必要がある。そのため各タスクは
自分が依存するステップの並びを持ち、ワークフローがそれを順に実行する。電荷密度は
QE の ``outdir`` を 1 つ共有することで使い回す。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ezcal.engines import get_engine
from ezcal.engines.base import CalcResult
from ezcal.scheduler import get_scheduler

#: タスク -> そのタスクの前に実行すべきステップ (実行順)
CHAINS: dict[str, tuple[str, ...]] = {
    "scf": ("scf",),
    "relax": ("relax",),
    "vc-relax": ("vc-relax",),
    "nscf": ("scf", "nscf"),
    "dos": ("scf", "nscf", "dos"),
    "pdos": ("scf", "nscf", "dos"),
    "bands": ("scf", "bands"),
    "charge": ("scf", "dos", "charge"),      # 先に dos: projwfc が Loewdin 電荷を出すため
    "auto": ("vc-relax", "scf", "nscf", "dos", "bands"),
}

STEP_TITLES = {
    "relax": "原子位置の最適化",
    "vc-relax": "セルと原子位置の最適化",
    "scf": "自己無撞着場計算",
    "nscf": "非自己無撞着計算 (密なメッシュ)",
    "dos": "状態密度",
    "bands": "バンド構造",
    "charge": "電荷密度と原子電荷",
}


@dataclass
class WorkflowResult:
    task: str
    rundir: Path
    steps: dict[str, CalcResult] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)
    exports: list[Path] = field(default_factory=list)
    structure_initial: Any = None
    structure_final: Any = None
    ok: bool = False
    submitted_only: bool = False
    elapsed: float = 0.0
    messages: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)

    @property
    def last(self) -> CalcResult | None:
        return self.steps[self.order[-1]] if self.order else None

    def summary(self) -> dict:
        from ezcal.structures import structure_info

        payload: dict[str, Any] = {
            "task": self.task,
            "ok": self.ok,
            "rundir": str(self.rundir),
            "elapsed_s": round(self.elapsed, 2),
            "steps": {name: self.steps[name].summary() for name in self.order},
            "plots": [str(p) for p in self.plots],
            "exports": [str(p) for p in self.exports],
            "messages": self.messages,
        }
        if self.structure_initial is not None:
            payload["structure_initial"] = structure_info(self.structure_initial)
        if self.structure_final is not None:
            payload["structure_final"] = structure_info(self.structure_final)
            payload["structure_final_cif"] = str(self.rundir / "final_structure.cif")
        return payload


class Workflow:
    """1 つの構造に対して、タスク (と、その前提となる処理すべて) を実行する。"""

    def __init__(self, config, structure, rundir: str | Path, label: str = "",
                 log: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.structure = self._apply_sublattices(config, structure,
                                                 log or (lambda msg: None))
        self.rundir = Path(rundir)
        self.label = label or structure.composition.reduced_formula
        self.log = log or (lambda msg: None)
        self.scheduler = get_scheduler(config)
        self.engine = get_engine(config.get("engine", "qe"), config, self.scheduler)
        self.outdir = (self.rundir / "tmp").resolve()

    @staticmethod
    def _apply_sublattices(config, structure, log):
        """``dft.magnetic_sublattices`` を pw.x の元素ラベルに変換する。

        ``{"Fe": [0.6, -0.6]}`` を指定すると、Fe サイトには出現順に Fe1、Fe2、
        Fe1、... とラベルが振られる。pw.x から見れば反強磁性体は 2 種類の元素と
        なり、それに応じて対称性が下がる。
        """
        from ezcal.structures import split_sublattices

        spec = config.get("dft.magnetic_sublattices") or {}
        spec = {element: values for element, values in spec.items() if values}
        if not spec:
            return structure

        labelled, magnetization = split_sublattices(structure, spec)
        config.set("dft.nspin", 2)
        merged = dict(config.get("dft.starting_magnetization") or {})
        merged.update(magnetization)
        config.set("dft.starting_magnetization", merged)
        summary = ", ".join(f"{label} {value:+g}"
                            for label, value in sorted(magnetization.items()))
        log(f"    磁気副格子: {summary}")
        return labelled

    # ------------------------------------------------------------ 実行計画
    def plan(self, task: str, skip: Sequence[str] = (), only: bool = False) -> list[str]:
        task = task.lower()
        chain = list(CHAINS.get(task, (task,)))
        if only:
            chain = [task] if task != "auto" else chain
        chain = [step for step in chain if step not in skip]
        return [step for step in chain if self.engine.supports(step)]

    # ---------------------------------------------------------------- 実行
    def run(self, task: str, skip: Sequence[str] = (), only: bool = False) -> WorkflowResult:
        start = time.time()
        result = WorkflowResult(task=task, rundir=self.rundir,
                                structure_initial=self.structure)
        if task not in CHAINS or task != "auto":
            wanted = task if task in CHAINS else task
            if wanted != "auto" and not self.engine.supports(wanted):
                result.messages.append(
                    f"エンジン {self.engine.name!r} は {wanted!r} を実行できません "
                    f"(対応タスク: {', '.join(self.engine.supported)})")
                result.elapsed = time.time() - start
                return result
        steps = self.plan(task, skip=skip, only=only)
        if task == "auto" and len(steps) < len(CHAINS["auto"]):
            dropped = [s for s in CHAINS["auto"] if s not in steps and s not in skip]
            if dropped:
                result.messages.append(
                    f"エンジン {self.engine.name!r} が未対応のステップを省略しました: "
                    + ", ".join(dropped))
        if not steps:
            result.messages.append(
                f"エンジン {self.engine.name!r} は {task!r} に必要なステップをどれも実行できません "
                f"(対応タスク: {', '.join(self.engine.supported)})"
            )
            result.elapsed = time.time() - start
            return result

        self.rundir.mkdir(parents=True, exist_ok=True)
        self._save_structure(self.structure, "input_structure")

        problems = self.engine.check()
        if problems:
            result.messages += problems
            result.elapsed = time.time() - start
            return result

        structure = self.structure
        kpath = None
        prev: CalcResult | None = None

        for index, step in enumerate(steps):
            # バンド経路はセルを固定してしまうため、scf の前に経路が基準とする
            # プリミティブセルへ切り替え、以降のステップ全体で共有する
            if "bands" in steps and step == "scf" and kpath is None:
                kpath, structure = self._prepare_band_path(structure)

            workdir = self.rundir / f"{index:02d}_{step}"
            self.log(f"[{index + 1}/{len(steps)}] {step}: {STEP_TITLES.get(step, step)}")
            kwargs: dict[str, Any] = {"outdir": self.outdir}
            if step == "nscf":
                kwargs["kmesh"] = self._nscf_mesh(structure)
                kwargs["occupations"] = self.config.get("nscf.occupations", "tetrahedra")
            if step == "bands" and kpath is not None:
                kwargs["kpath"] = kpath

            step_result = self.engine.run(structure, step, workdir, prev=prev, **kwargs)
            result.steps[step] = step_result
            result.order.append(step)
            step_result.save_json()
            for note in step_result.messages:
                result.messages.append(f"[{step}] {note}")

            if step_result.submitted_only:
                result.submitted_only = True
                result.ok = True
                if step_result.job_id:
                    result.messages.append(
                        f"ステップ {step!r} をジョブ {step_result.job_id} として投入しました。"
                        "残りのステップも自動で続けるには --qsub-wait を付けてください")
                else:
                    remaining = ", ".join(steps[index + 1:]) or "なし"
                    result.messages.append(
                        f"ドライラン: {step!r} の入力ファイルとジョブスクリプトを書き出しました。"
                        f"実行は行っていません (残りのステップ: {remaining})")
                break
            if not step_result.ok:
                result.messages.append(f"ステップ {step!r} が失敗しました")
                break

            if step_result.structure is not None and step in {"relax", "vc-relax"}:
                structure = step_result.structure
                self._save_structure(structure, "relaxed_structure")
            prev = step_result
        else:
            result.ok = True

        result.structure_final = structure
        if result.structure_final is not None:
            self._save_structure(result.structure_final, "final_structure")
        result.elapsed = time.time() - start

        if not result.submitted_only:
            self._postprocess(result)
        self._write_summary(result)
        self._cleanup()
        return result

    # ----------------------------------------------------------- 内部処理
    def _prepare_band_path(self, structure):
        from ezcal.structures import band_path

        kpath = band_path(
            structure,
            line_density=float(self.config.get("bands.line_density", 25)),
            symprec=float(self.config.get("bands.symprec", 1e-5)),
            min_points=int(self.config.get("bands.min_points_per_segment", 6)),
            scheme=self.config.get("bands.scheme"),
        )
        primitive = kpath.primitive_structure
        if primitive is not None and len(primitive) != len(structure):
            self.log(f"    バンド経路: 標準プリミティブセルを使用します "
                     f"({len(structure)} -> {len(primitive)} 原子)")
        path_text = " -> ".join(dict.fromkeys(
            [lab for _, lab in kpath.labels if lab]))
        self.log(f"    k 経路 ({kpath.scheme}, {kpath.nkpt} 点): {path_text}")
        return kpath, primitive if primitive is not None else structure

    def _nscf_mesh(self, structure) -> list[int]:
        from ezcal.structures import scale_kmesh

        base = self.engine._kmesh(self.config, structure)
        return scale_kmesh(base, float(self.config.get("nscf.kmesh_scale", 2)))

    def _save_structure(self, structure, name: str) -> None:
        try:
            structure.to(filename=str(self.rundir / f"{name}.cif"))
            (self.rundir / f"{name}.json").write_text(
                json.dumps(structure.as_dict(), indent=1, default=str), encoding="utf-8")
        except Exception as exc:                       # これが原因で計算を失敗させない
            self.log(f"    ({name} を書き出せませんでした: {exc})")

    def _postprocess(self, result: WorkflowResult) -> None:
        from ezcal import plotting

        backends = plotting.resolve_backends(self.config.get("output.plot"))
        if not backends:
            return
        plots_dir = self.rundir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        dpi = int(self.config.get("output.dpi", 200))

        fermi = None
        zero = "F"
        for name in ("bands", "nscf", "scf"):
            step = result.steps.get(name)
            if step is None:
                continue
            gap_info = step.data.get("gap_info") or {}
            if gap_info and not gap_info.get("metal") and gap_info.get("vbm") is not None:
                fermi = gap_info["vbm"]      # 価電子帯上端をゼロ点にする
                zero = "VBM"
                break
            if step.fermi_energy is not None:
                fermi = step.fermi_energy
                break

        bands = result.steps.get("bands")
        dos = result.steps.get("dos") or next(
            (result.steps[name] for name in reversed(result.order)
             if result.steps[name].ok and result.steps[name].data.get("dos")), None)
        self._save_raw(bands, dos, fermi, zero)
        emin = self.config.get("bands.emin")
        emax = self.config.get("bands.emax")
        emin = float(emin) if emin is not None else -10.0
        emax = float(emax) if emax is not None else 10.0

        if bands is not None and bands.ok and "eigenvalues" in bands.data:
            result.plots += plotting.plot_bands(
                bands, plots_dir, backends, fermi=fermi, emin=emin, emax=emax,
                dpi=dpi, title=f"{self.label} band structure", zero=zero,
                plotter=self.config.get("bands.plotter", "auto"))
            result.exports.append(
                plotting.export_bands_csv(bands, plots_dir / "bands.csv", fermi))
        if dos is not None and dos.ok:
            result.plots += plotting.plot_dos(
                dos, plots_dir, backends, fermi=fermi, emin=emin, emax=emax,
                dpi=dpi, title=f"{self.label} density of states", zero=zero)
            csv = plotting.export_dos_csv(dos, plots_dir / "dos.csv", fermi)
            if csv:
                result.exports.append(csv)
        if bands is not None and dos is not None and bands.ok and dos.ok:
            result.plots += plotting.plot_bands_dos(
                bands, dos, plots_dir, backends, fermi=fermi, emin=emin, emax=emax,
                dpi=dpi, title=self.label, zero=zero)
        result.plots += plotting.plot_convergence(result.steps, plots_dir, backends, dpi)
        self._plot_charge(result, plots_dir, backends, dpi)

    def _plot_charge(self, result: WorkflowResult, plots_dir: Path,
                     backends: Sequence[str], dpi: int) -> None:
        """書き出された各 cube について、断面プロファイル・断面図・3D 等値面を描く。"""
        from ezcal import plotting
        from ezcal.charge import read_cube

        step = result.steps.get("charge")
        if step is None or not step.ok:
            return
        raw_axis = self.config.get("charge.slice_axis")
        axis = None if raw_axis is None else int(raw_axis)
        fraction = float(self.config.get("charge.slice_fraction", 0.5))
        for key, path in step.files.items():
            if not key.startswith("cube_"):
                continue
            kind = key[len("cube_"):]
            try:
                cube = read_cube(path)
            except Exception as exc:
                result.messages.append(f"{kind} を描画できませんでした: {exc}")
                continue
            label = f"{self.label} {kind}"
            result.plots += plotting.plot_charge_profile(
                cube, plots_dir, backends, kind=kind, dpi=dpi, title=label)
            result.plots += plotting.plot_charge_slice(
                cube, plots_dir, backends, kind=kind, axis=axis, fraction=fraction,
                dpi=dpi, title=label)
            if self.config.get("charge.isosurface", True):
                levels = self.config.get("charge.isosurface_levels")
                result.plots += plotting.plot_charge_isosurface(
                    cube, plots_dir, kind=kind, backends=backends,
                    max_points=int(self.config.get("charge.isosurface_grid", 64)),
                    levels=levels, dpi=dpi,
                    opacity=float(self.config.get("charge.isosurface_opacity", 0.45)),
                    title=label)
            if not self.config.get("charge.keep_cube", True):
                Path(path).unlink(missing_ok=True)
        self._plot_charge_map(result, step, plots_dir, backends, dpi)

    def _plot_charge_map(self, result: WorkflowResult, step, plots_dir: Path,
                         backends: Sequence[str], dpi: int) -> None:
        """原子ごとの価数を 3D 空間にマッピングする (色 = 電荷、大きさ = |電荷|)。"""
        from ezcal import plotting

        if not self.config.get("charge.map3d", True):
            return
        rows = step.data.get("atoms") or []
        if not rows:
            return
        lattice = None
        structure = result.structure_final or result.structure_initial
        if structure is not None:
            lattice = structure.lattice.matrix
        try:
            result.plots += plotting.plot_charge_map_3d(
                rows, plots_dir, backends, lattice=lattice,
                source=str(self.config.get("charge.map_source", "auto")), dpi=dpi,
                title=f"{self.label} atomic charges")
        except Exception as exc:
            result.messages.append(f"原子電荷の 3D マッピングを描けませんでした: {exc}")

    def _save_raw(self, bands, dos, fermi, zero: str = "F") -> None:
        """バンド/DOS の配列を保存し、`ezcal plot` が QE 抜きで再描画できるようにする。"""
        import numpy as np

        payload: dict[str, Any] = {"fermi": fermi, "zero": zero}
        if bands is not None and bands.ok and bands.data.get("eigenvalues") is not None:
            payload["bands"] = {
                "eigenvalues": np.asarray(bands.data["eigenvalues"]).tolist(),
                "kpath": bands.data.get("kpath", {}),
            }
        if dos is not None and dos.ok and dos.data.get("dos"):
            raw = dos.data["dos"]
            payload["dos"] = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") else v)
                              for k, v in raw.items() if v is not None}
            pdos = dos.data.get("pdos") or {}
            if pdos:
                payload["pdos"] = {
                    key: ({k: np.asarray(v).tolist() for k, v in value.items()}
                          if isinstance(value, dict)
                          else (np.asarray(value).tolist()
                                if hasattr(value, "__len__") else value))
                    for key, value in pdos.items() if value is not None
                }
        if len(payload) > 1:
            (self.rundir / "raw_data.json").write_text(
                json.dumps(payload), encoding="utf-8")

    def _write_properties(self, result: WorkflowResult) -> dict:
        """物理量を 1 箇所に集約する。

        バンドギャップ、フェルミ準位、原子ごとの磁気モーメントや電荷は複数の
        ステップに分散している。ここでまとめておけば、どのステップがどの値を
        出したのかを利用者が知る必要はなくなる。
        """
        from ezcal.structures import site_labels

        dense = next((result.steps[n] for n in ("bands", "nscf", "scf")
                      if n in result.steps and result.steps[n].ok), None)
        scf = result.steps.get("scf") or (result.steps.get(result.order[0])
                                          if result.order else None)
        charge = result.steps.get("charge")

        payload: dict[str, Any] = {"formula": None, "task": result.task,
                                   "rundir": str(self.rundir)}
        if result.structure_final is not None:
            payload["formula"] = result.structure_final.composition.reduced_formula

        if dense is not None:
            gap_info = dense.data.get("gap_info") or {}
            payload["electronic"] = {
                "fermi_energy_eV": dense.fermi_energy,
                "band_gap_eV": dense.band_gap,
                "metal": gap_info.get("metal"),
                "gap_kind": None if gap_info.get("metal") in (None, True) else
                            ("direct" if gap_info.get("direct") else "indirect"),
                "vbm_eV": gap_info.get("vbm"),
                "cbm_eV": gap_info.get("cbm"),
                "nelec": dense.nelec,
                "nbnd": dense.nbnd,
                "source_step": dense.task,
            }
        if scf is not None:
            payload["magnetism"] = {
                "nspin": int(self.config.get("dft.nspin", 1) or 1),
                "total_magnetization_uB": scf.magnetization,
                "absolute_magnetization_uB": scf.abs_magnetization,
            }
        if result.structure_final is not None:
            labels = site_labels(result.structure_final)
            moments = (scf.site_magnetization if scf is not None else None) or []
            charge_rows = {row["index"]: row for row in
                           ((charge.data.get("atoms") if charge else None) or [])}
            atoms = []
            for index, site in enumerate(result.structure_final):
                entry: dict[str, Any] = {
                    "index": index + 1, "label": labels[index],
                    "element": site.specie.symbol,
                    "moment_sphere_uB": moments[index] if index < len(moments) else None,
                }
                extra = charge_rows.get(index + 1, {})
                for key, out in (("lowdin_moment", "moment_lowdin_uB"),
                                 ("lowdin_charge", "lowdin_charge_e"),
                                 ("lowdin_electrons", "lowdin_electrons"),
                                 ("bader_charge", "bader_charge_e"),
                                 ("bader_electrons", "bader_electrons"),
                                 ("bader_volume", "bader_volume_A3")):
                    if extra.get(key) is not None:
                        entry[out] = extra[key]
                atoms.append(entry)
            payload["atoms"] = atoms

        (self.rundir / "properties.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (self.rundir / "properties.md").write_text(
            render_properties(payload), encoding="utf-8")
        return payload

    def _write_summary(self, result: WorkflowResult) -> None:
        try:
            result.properties = self._write_properties(result)
        except Exception as exc:                      # サマリのせいで計算結果を失わない
            self.log(f"    (properties を書き出せませんでした: {exc})")
        payload = result.summary()
        (self.rundir / "summary.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        (self.rundir / "report.md").write_text(render_report(result, self.config),
                                               encoding="utf-8")

    def _cleanup(self) -> None:
        if self.config.get("output.keep_wavefunctions", False):
            return
        save = self.outdir
        if not save.is_dir():
            return
        patterns = ("wfc*.dat", "*.wfc*", "*.igk*")
        for path in [q for pattern in patterns for q in save.rglob(pattern)]:
            try:
                path.unlink()
            except OSError:
                pass


# -------------------------------------------------------------- レポート出力
def render_report(result: WorkflowResult, config) -> str:
    from ezcal.structures import structure_info

    lines = [f"# ezcal レポート - {result.task}", ""]
    lines.append(f"- 実行ディレクトリ: `{result.rundir}`")
    lines.append(f"- 状態: {'成功' if result.ok else '失敗'}")
    lines.append(f"- 実時間: {result.elapsed:.1f} 秒")
    lines.append(f"- エンジン: `{config.get('engine', 'qe')}`  "
                 f"スケジューラ: `{config.get('run.scheduler', 'local')}`  "
                 f"プロセス数: {config.get('run.nproc')}")
    lines.append("")

    if result.structure_initial is not None:
        info = structure_info(result.structure_initial)
        lines += ["## 構造 (入力)", "",
                  f"- 組成式: **{info['formula']}**  ({info['natoms']} 原子)",
                  f"- 空間群: {info.get('spacegroup')} (#{info.get('spacegroup_number')})",
                  f"- a, b, c = {info['lattice']['a']:.4f}, {info['lattice']['b']:.4f}, "
                  f"{info['lattice']['c']:.4f} A",
                  f"- 体積: {info['volume']:.3f} A^3", ""]

    lines += ["## 計算条件", "",
              f"- ecutwfc / ecutrho: {config.get('dft.ecutwfc')} / "
              f"{config.get('dft.ecutrho')} Ry",
              f"- 汎関数: {config.get('dft.functional')}",
              f"- 占有数: {config.get('dft.occupations')} "
              f"({config.get('dft.smearing')}, degauss={config.get('dft.degauss')} Ry)",
              f"- nspin: {config.get('dft.nspin')}", ""]

    magnetic = any(result.steps[n].magnetization is not None for n in result.order)
    header = ["ステップ", "成否", "エネルギー (eV)", "E/原子 (eV)", "E_F (eV)", "ギャップ (eV)",
              "max\\|F\\| (eV/A)", "P (GPa)"]
    if magnetic:
        header += ["M (uB/cell)", "|M| (uB/cell)"]
    header.append("時間 (秒)")
    lines += ["## 各ステップ", "",
              "| " + " | ".join(header) + " |",
              "|" + "---|" * len(header)]
    for name in result.order:
        step = result.steps[name]

        def fmt(value, digits=4):
            return "-" if value is None else f"{value:.{digits}f}"

        row = [name, "成功" if step.ok else "失敗", fmt(step.energy, 6),
               fmt(step.energy_per_atom, 6), fmt(step.fermi_energy),
               fmt(step.band_gap, 3), fmt(step.max_force, 4), fmt(step.pressure, 2)]
        if magnetic:
            row += [fmt(step.magnetization, 3), fmt(step.abs_magnetization, 3)]
        row.append(fmt(step.walltime, 1))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    site_moments = next((result.steps[n].site_magnetization for n in result.order
                         if result.steps[n].site_magnetization), None)
    if site_moments:
        from ezcal.structures import site_labels

        names = (site_labels(result.structure_final)
                 if result.structure_final is not None else
                 [str(i + 1) for i in range(len(site_moments))])
        pairs = ", ".join(f"{name} {value:+.3f}"
                          for name, value in zip(names, site_moments))
        lines += [f"サイトごとの磁気モーメント (uB): {pairs}", ""]
    if result.order:
        gap_info = next((result.steps[n].data.get("gap_info") for n in reversed(result.order)
                         if result.steps[n].data.get("gap_info")), None)
        if gap_info:
            if gap_info.get("metal"):
                lines += ["固有値スペクトルにギャップがありません: **金属的**。", ""]
            else:
                kind = "直接" if gap_info.get("direct") else "間接"
                lines += [f"バンドギャップ: **{gap_info['gap']:.3f} eV** ({kind}遷移)、"
                          f"VBM {gap_info['vbm']:.3f} eV、CBM {gap_info['cbm']:.3f} eV。"
                          "図は VBM を基準にしています。", ""]

    if result.plots:
        lines += ["## 図", ""]
        for path in result.plots:
            lines.append(f"- `{path}`")
        lines.append("")
    if result.exports:
        lines += ["## データ", ""] + [f"- `{p}`" for p in result.exports] + [""]
    if result.messages:
        lines += ["## 備考", ""] + [f"- {m}" for m in result.messages] + [""]
    return "\n".join(lines)


def render_properties(payload: Mapping[str, Any]) -> str:
    """「この計算で何が得られたのか」に 1 ページで答えるための出力。"""
    def fmt(value, digits=4, unit=""):
        if value is None:
            return "-"
        return f"{value:.{digits}f}{unit}"

    lines = [f"# {payload.get('formula') or '物性値'}", "",
             f"- 実行: `{payload.get('rundir')}`  (タスク `{payload.get('task')}`)", ""]

    electronic = payload.get("electronic") or {}
    if electronic:
        metal = electronic.get("metal")
        gap = electronic.get("band_gap_eV")
        lines += ["## 電子構造", "", "| 物理量 | 値 |", "|---|---|",
                  f"| フェルミエネルギー | {fmt(electronic.get('fermi_energy_eV'))} eV |"]
        if metal:
            lines.append("| バンドギャップ | **0** (金属的: バンドがフェルミ準位を横切る) |")
        else:
            kind = electronic.get("gap_kind")
            kind_ja = {"direct": "直接遷移", "indirect": "間接遷移"}.get(kind, kind)
            lines.append(f"| バンドギャップ | **{fmt(gap, 4)} eV**"
                         f"{f' ({kind_ja})' if kind else ''} |")
            lines.append(f"| 価電子帯上端 (VBM) | {fmt(electronic.get('vbm_eV'))} eV |")
            lines.append(f"| 伝導帯下端 (CBM) | {fmt(electronic.get('cbm_eV'))} eV |")
        lines += [f"| セル内の電子数 | {fmt(electronic.get('nelec'), 1)} |",
                  f"| 計算したバンド数 | {electronic.get('nbnd') or '-'} |",
                  f"| 取得元 | `{electronic.get('source_step')}` ステップ |", ""]

    magnetism = payload.get("magnetism") or {}
    if magnetism.get("nspin") == 2:
        lines += ["## 磁性", "",
                  f"- 全磁化: **{fmt(magnetism.get('total_magnetization_uB'), 3)}** "
                  "μB/cell",
                  f"- 絶対磁化: **{fmt(magnetism.get('absolute_magnetization_uB'), 3)}** "
                  "μB/cell  (反強磁性体で意味を持つのはこちら)", ""]

    atoms = payload.get("atoms") or []
    if atoms:
        columns = [("index", "#", 0), ("label", "サイト", None), ("element", "元素", None),
                   ("moment_sphere_uB", "磁気モーメント (μB)", 3),
                   ("moment_lowdin_uB", "Löwdin モーメント (μB)", 3),
                   ("lowdin_charge_e", "Löwdin 電荷 (e)", 3),
                   ("bader_charge_e", "Bader 電荷 (e)", 3),
                   ("bader_volume_A3", "Bader 体積 (Å³)", 2)]
        present = [c for c in columns
                   if c[0] in {"index", "label", "element"}
                   or any(a.get(c[0]) is not None for a in atoms)]
        lines += ["## 原子ごとの値", "",
                  "| " + " | ".join(c[1] for c in present) + " |",
                  "|" + "---|" * len(present)]
        for atom in atoms:
            cells = []
            for key, _, digits in present:
                value = atom.get(key)
                cells.append("-" if value is None else
                             (str(value) if digits is None else fmt(value, digits)))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        if any(a.get("bader_charge_e") is not None for a in atoms):
            lines += ["Bader 電荷は、その原子から電子が奪われている場合に正の値になります。"
                      "FFT グリッド上で電荷密度を分割して求めるため、グリッドの粗さに由来する "
                      "0.01〜0.05 e 程度の不確かさを含みます。", ""]
    return "\n".join(lines)
