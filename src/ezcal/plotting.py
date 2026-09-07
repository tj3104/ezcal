"""バンド構造、状態密度、構造最適化の履歴の作図。

どの関数も ``backends`` のリスト (``matplotlib``、``plotly``、``both``、
``none`` のいずれか) を受け取り、書き出したファイルの一覧を返す。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SPIN_LABEL = {0: "up", 1: "down"}
_ORBITAL_ORDER = "spdf"


def resolve_backends(value: Any) -> list[str]:
    if value is None:
        return ["matplotlib"]
    if isinstance(value, str):
        value = [value]
    out: list[str] = []
    for item in value:
        item = str(item).lower()
        if item in {"none", "off"}:
            return []
        if item == "both":
            out += ["matplotlib", "plotly"]
        elif item in {"matplotlib", "mpl", "png"}:
            out.append("matplotlib")
        elif item in {"plotly", "html"}:
            out.append("plotly")
    return list(dict.fromkeys(out)) or ["matplotlib"]


def _align(values, source_energy, target_energy) -> np.ndarray | None:
    """PDOS のチャンネルを DOS のエネルギーグリッドに載せ替える (両者はグリッドが異なる)。"""
    values = np.asarray(values, dtype=float)
    target_energy = np.asarray(target_energy, dtype=float)
    if source_energy is None:
        return values if values.shape == target_energy.shape else None
    source_energy = np.asarray(source_energy, dtype=float)
    if values.shape != source_energy.shape:
        return None
    if values.shape == target_energy.shape and np.allclose(source_energy, target_energy):
        return values
    return np.interp(target_energy, source_energy, values, left=0.0, right=0.0)


def _pdos_channels(pdos: Mapping[str, Any] | None, key: str,
                   target_energy: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if not pdos or not pdos.get(key):
        return []
    source = pdos.get("energy")
    out: list[tuple[str, np.ndarray]] = []
    for name in sorted(pdos[key],
                       key=lambda k: (k.split("-")[0], _ORBITAL_ORDER.find(k[-1]))):
        aligned = _align(pdos[key][name], source, target_energy)
        if aligned is not None:
            out.append((name, aligned))
    return out


_PALETTE = ["#e07b39", "#2e8b57", "#8e44ad", "#c0392b", "#16a085", "#d4ac0d",
            "#2980b9", "#7f8c8d"]


def _channel_colours(channels) -> dict:
    """軌道ごとに 1 色を割り当て、up 成分と down 成分で同じ色を共有する。"""
    colours, order = {}, []
    for name, _ in channels:
        base = name.rsplit(" ", 1)[0] if name.endswith((" up", " down")) else name
        if base not in order:
            order.append(base)
        colours[name] = _PALETTE[order.index(base) % len(_PALETTE)]
    return colours


def energy_axis_label(zero: str = "F", latex: bool = True) -> str:
    """エネルギーの基準点を示す軸ラベル (金属なら E_F、それ以外は VBM)。"""
    if latex:
        return rf"$E - E_\mathrm{{{zero}}}$  (eV)"
    return f"E - E_{zero} (eV)"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11,
        "axes.linewidth": 1.0,
        "figure.autolayout": False,
    })
    return plt


# ------------------------------------------------------------ バンドデータ
def band_arrays(bands_result, fermi: float | None = None) -> dict:
    """バンドの :class:`CalcResult` を素の配列に整形する。"""
    data = bands_result.data
    eig = np.asarray(data["eigenvalues"])                # (nspin, nk, nbnd)
    kpath = data.get("kpath", {})
    distances = np.asarray(kpath.get("distances", np.arange(eig.shape[1])), dtype=float)
    labels = [(int(i), str(lab)) for i, lab in kpath.get("labels", [])]
    ef = fermi if fermi is not None else (bands_result.fermi_energy or 0.0)
    kpoints = kpath.get("kpoints")
    return {"eigenvalues": eig, "distances": distances, "labels": labels, "fermi": ef,
            "kpoints": None if kpoints is None else np.asarray(kpoints, dtype=float)}


#: バンド図の描画器。auto は materials_project 経路なら BSPlotter を選ぶ
BAND_PLOTTERS = ("auto", "bsplotter", "ezcal")


#: pymatgen で切った経路。BSPlotter がそのまま扱える
_PYMATGEN_SCHEMES = frozenset({"materials_project", "latimer_munro", "setyawan_curtarolo"})


def resolve_band_plotter(bands_result, plotter: str | None = "auto") -> str:
    """``auto`` を実際の描画器名へ落とす。

    pymatgen で切った経路 (既定の Materials Project 方式を含む) は pymatgen の
    ``BSPlotter`` で描く。seekpath 経路と、逆格子情報を持たない古い実行結果は、
    ezcal 自前の描画にフォールバックする。
    """
    name = str(plotter or "auto").strip().lower()
    if name not in BAND_PLOTTERS:
        raise ValueError(f"未知のバンド描画器 {plotter!r} です。"
                         f"使えるのは {', '.join(BAND_PLOTTERS)} です")
    kpath = (getattr(bands_result, "data", None) or {}).get("kpath") or {}
    usable = bool(kpath.get("reciprocal_lattice")) and bool(kpath.get("kpoints"))
    if name == "bsplotter":
        return "bsplotter" if usable else "ezcal"
    if name == "auto":
        if usable and str(kpath.get("scheme", "")) in _PYMATGEN_SCHEMES:
            return "bsplotter"
        return "ezcal"
    return "ezcal"


def band_structure_symm_line(bands_result, fermi: float | None = None):
    """ezcal のバンド結果から pymatgen の ``BandStructureSymmLine`` を組む。

    ``fermi`` はエネルギーの基準点 (金属なら E_F、絶縁体なら VBM)。BSPlotter の
    ``zero_to_efermi`` がこの値を 0 に合わせるので、ezcal 自前の描画と同じ縦軸に
    なる。
    """
    from pymatgen.core import Lattice
    from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
    from pymatgen.electronic_structure.core import Spin

    data = bands_result.data
    kpath = data.get("kpath") or {}
    recip = kpath.get("reciprocal_lattice")
    kpoints = kpath.get("kpoints")
    if not recip or not kpoints:
        raise ValueError("BSPlotter には k 経路の逆格子と k 点座標が必要です "
                         "(この結果は古い形式で保存されています)")

    eig = np.asarray(data["eigenvalues"])                  # (nspin, nk, nbnd)
    ef = fermi if fermi is not None else (bands_result.fermi_energy or 0.0)
    spins = (Spin.up, Spin.down)
    # pymatgen は (nbnd, nk) の並びを取る
    eigenvals = {spins[s]: eig[s].T for s in range(eig.shape[0])}

    kpoints = np.asarray(kpoints, dtype=float)
    labels_dict: dict[str, list[float]] = {}
    raw = kpath.get("raw_labels") or kpath.get("labels") or []
    for index, name in raw:
        index = int(index)
        if not (0 <= index < len(kpoints)):
            continue
        # 経路の切れ目で 2 つのラベルが同じ点に乗ることがある ("U|K")
        for part in str(name).split("|"):
            part = part.strip()
            if part:
                labels_dict.setdefault(part, kpoints[index].tolist())

    return BandStructureSymmLine(
        kpoints, eigenvals, Lattice(np.asarray(recip, dtype=float)), float(ef),
        labels_dict, coords_are_cartesian=False,
        structure=getattr(bands_result, "structure", None),
    )


def _plot_bands_bsplotter(bands_result, path: Path, fermi: float | None,
                          lo: float, hi: float, dpi: int, title: str, zero: str) -> Path:
    """pymatgen の BSPlotter でバンド図を描く (Materials Project と同じ見た目)。"""
    from pymatgen.electronic_structure.plotter import BSPlotter

    plt = _mpl()
    bs = band_structure_symm_line(bands_result, fermi)
    ax = BSPlotter(bs).get_plot(zero_to_efermi=True, ylim=(lo, hi))
    ax.set_ylabel(energy_axis_label(zero))
    ax.set_title(title)
    legend = ax.get_legend()
    if legend is not None and not bs.is_spin_polarized:
        # スピン非分極では "Band 0 up" の凡例は何も足さない
        legend.remove()
    fig = ax.get_figure()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _path_breaks(distances: np.ndarray, kpoints: np.ndarray | None = None) -> np.ndarray:
    """バンド経路の切れ目 (``U|K`` など) の直後のインデックス。

    :func:`ezcal.structures.band_path` は区間ごとに独立してサンプリングするため、
    経路長が進まない箇所が 2 種類ある。区間の「つなぎ目」(同じ k 点が 2 度並ぶ)
    と、経路の「切れ目」(別の k 点が同じ経路長に並ぶ) で、線を切るのは後者だけ。
    ``kpoints`` を渡さない場合は経路長だけで判定する。
    """
    distances = np.asarray(distances, dtype=float)
    if distances.size < 2:
        return np.empty(0, dtype=int)
    span = float(distances[-1] - distances[0]) or 1.0
    stalled = np.diff(distances) <= span * 1e-9
    if kpoints is not None:
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.shape[0] == distances.size:
            moved = np.abs(np.diff(kpoints, axis=0)).max(axis=1) > 1e-8
            stalled &= moved
    return np.flatnonzero(stalled) + 1


def _broken(values: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """切れ目に NaN を挟み、線が跨いで繋がらないようにする。"""
    values = np.asarray(values, dtype=float)
    if breaks.size == 0:
        return values
    return np.insert(values, breaks, np.nan)


def _tick_positions(distances: np.ndarray, labels: Sequence[tuple[int, str]]):
    """同じ経路長に位置するラベルをまとめる (バンド経路の切れ目の処理)。"""
    grouped: dict[int, tuple[float, str]] = {}
    for index, name in labels:
        if not (0 <= index < len(distances)) or not name:
            continue
        position = float(distances[index])
        # 経路長でまとめるが、値そのものは丸めずに保持する。丸めた目盛りは軸の
        # 範囲からわずかに外れることがあり、matplotlib に捨てられてしまう
        key = int(round(position * 1e6))
        if key not in grouped:
            grouped[key] = (position, name)
        else:
            _, previous = grouped[key]
            if name not in previous.split("|"):
                grouped[key] = (position, f"{previous}|{name}")
    ordered = [grouped[k] for k in sorted(grouped)]
    return [pos for pos, _ in ordered], [name for _, name in ordered]


# ------------------------------------------------------------------ バンド
def plot_bands(bands_result, outdir: str | Path, backends: Iterable[str] = ("matplotlib",),
               fermi: float | None = None, emin: float | None = None,
               emax: float | None = None, dpi: int = 200,
               title: str = "band structure", zero: str = "F",
               plotter: str | None = "auto") -> list[Path]:
    arrays = band_arrays(bands_result, fermi)
    eig, dist = arrays["eigenvalues"], arrays["distances"]
    ef = arrays["fermi"]
    ticks, names = _tick_positions(dist, arrays["labels"])
    breaks = _path_breaks(dist, arrays.get("kpoints"))
    xdist = _broken(dist, breaks)
    lo = emin if emin is not None else -10.0
    hi = emax if emax is not None else 10.0
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    chosen = resolve_band_plotter(bands_result, plotter)

    for backend in backends:
        if backend == "matplotlib" and chosen == "bsplotter":
            try:
                written.append(_plot_bands_bsplotter(
                    bands_result, outdir / "bands.png", fermi, lo, hi, dpi, title, zero))
                continue
            except Exception as exc:      # 古い結果や特殊な経路では自前描画に戻す
                warnings.warn(f"BSPlotter で描けませんでした ({exc})。"
                              "ezcal 自前の描画に切り替えます", RuntimeWarning)

        if backend == "matplotlib":
            plt = _mpl()
            fig, ax = plt.subplots(figsize=(6.0, 4.6))
            colors = ["#1f4e9c", "#c0392b"]
            for spin in range(eig.shape[0]):
                for band in range(eig.shape[2]):
                    ax.plot(xdist, _broken(eig[spin, :, band] - ef, breaks), lw=1.1,
                            color=colors[spin % 2],
                            label=(f"spin {SPIN_LABEL[spin]}"
                                   if band == 0 and eig.shape[0] > 1 else None))
            ax.axhline(0.0, color="0.4", ls="--", lw=0.9)
            for tick in ticks:
                ax.axvline(tick, color="0.75", lw=0.7)
            ax.set_xticks(ticks)
            ax.set_xticklabels(names)
            ax.set_xlim(dist.min(), dist.max())
            ax.set_ylim(lo, hi)
            ax.set_ylabel(energy_axis_label(zero))
            ax.set_title(title)
            if eig.shape[0] > 1:
                ax.legend(frameon=False, loc="upper right", fontsize=9)
            fig.tight_layout()
            path = outdir / "bands.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            colors = ["#1f4e9c", "#c0392b"]
            for spin in range(eig.shape[0]):
                for band in range(eig.shape[2]):
                    fig.add_trace(go.Scatter(
                        x=xdist, y=_broken(eig[spin, :, band] - ef, breaks), mode="lines",
                        line=dict(width=1.4, color=colors[spin % 2]),
                        name=f"spin {SPIN_LABEL[spin]}" if band == 0 else None,
                        showlegend=(band == 0 and eig.shape[0] > 1),
                        hovertemplate="E-E_F = %{y:.3f} eV<extra></extra>",
                    ))
            fig.add_hline(y=0.0, line=dict(color="grey", dash="dash", width=1))
            for tick in ticks:
                fig.add_vline(x=tick, line=dict(color="lightgrey", width=1))
            fig.update_layout(
                title=title, template="plotly_white",
                xaxis=dict(tickvals=ticks, ticktext=[_plain(n) for n in names],
                           range=[float(dist.min()), float(dist.max())]),
                yaxis=dict(title=energy_axis_label(zero, latex=False), range=[lo, hi]),
                width=760, height=520,
            )
            path = outdir / "bands.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written


def _plain(label: str) -> str:
    return label.replace("$_{", "").replace("}$", "").replace("$", "")


# -------------------------------------------------------------- 状態密度
def plot_dos(dos_result, outdir: str | Path, backends: Iterable[str] = ("matplotlib",),
             fermi: float | None = None, emin: float | None = None,
             emax: float | None = None, dpi: int = 200,
             title: str = "density of states", zero: str = "F") -> list[Path]:
    data = dos_result.data.get("dos")
    if not data:
        return []
    energy = np.asarray(data["energy"])
    ef = fermi if fermi is not None else (data.get("fermi_energy") or 0.0)
    shifted = energy - ef
    lo = emin if emin is not None else -10.0
    hi = emax if emax is not None else 10.0
    mask = (shifted >= lo) & (shifted <= hi)
    pdos = dos_result.data.get("pdos") or {}
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    spin_pdos = bool(pdos.get("spin_polarised")) and bool(pdos.get("per_orbital_up"))
    if spin_pdos:
        channels = [(f"{name} up", values)
                    for name, values in _pdos_channels(pdos, "per_orbital_up", energy)]
        channels += [(f"{name} down", -values)
                     for name, values in _pdos_channels(pdos, "per_orbital_down", energy)]
    else:
        channels = _pdos_channels(pdos, "per_orbital", energy)

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, ax = plt.subplots(figsize=(6.2, 4.4))
            if "dos_up" in data:
                ax.plot(shifted[mask], np.asarray(data["dos_up"])[mask], color="#1f4e9c",
                        lw=1.4, label="total up")
                ax.plot(shifted[mask], -np.asarray(data["dos_down"])[mask], color="#c0392b",
                        lw=1.4, label="total down")
                ax.axhline(0, color="0.6", lw=0.8)
            else:
                ax.fill_between(shifted[mask], np.asarray(data["dos"])[mask],
                                color="0.85", label="total")
                ax.plot(shifted[mask], np.asarray(data["dos"])[mask], color="0.35", lw=1.2)
            colours = _channel_colours(channels)
            for name, values in channels:
                ax.plot(shifted[mask], values[mask], lw=1.2, color=colours[name],
                        label=name[:-3] if name.endswith(" up") else
                        (None if name.endswith(" down") else name))
            ax.axvline(0.0, color="0.4", ls="--", lw=0.9)
            ax.set_xlim(lo, hi)
            ax.set_xlabel(energy_axis_label(zero))
            ax.set_ylabel("DOS (states/eV)")
            ax.set_title(title)
            ax.legend(frameon=False, fontsize=9, ncol=2)
            fig.tight_layout()
            path = outdir / "dos.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            if "dos_up" in data:
                fig.add_trace(go.Scatter(x=shifted[mask], y=np.asarray(data["dos_up"])[mask],
                                         name="total up", line=dict(color="#1f4e9c")))
                fig.add_trace(go.Scatter(x=shifted[mask], y=-np.asarray(data["dos_down"])[mask],
                                         name="total down", line=dict(color="#c0392b")))
            else:
                fig.add_trace(go.Scatter(x=shifted[mask], y=np.asarray(data["dos"])[mask],
                                         name="total", fill="tozeroy",
                                         line=dict(color="#444444")))
            colours = _channel_colours(channels)
            for name, values in channels:
                fig.add_trace(go.Scatter(
                    x=shifted[mask], y=values[mask],
                    name=name[:-3] if name.endswith(" up") else name,
                    line=dict(color=colours[name]),
                    showlegend=not name.endswith(" down")))
            fig.add_vline(x=0.0, line=dict(color="grey", dash="dash"))
            fig.update_layout(title=title, template="plotly_white",
                              xaxis_title=energy_axis_label(zero, latex=False),
                              yaxis_title="DOS (states/eV)",
                              width=820, height=500)
            path = outdir / "dos.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written


# ---------------------------------------------------------- バンド + 状態密度
def plot_bands_dos(bands_result, dos_result, outdir: str | Path,
                   backends: Iterable[str] = ("matplotlib",), fermi: float | None = None,
                   emin: float = -10.0, emax: float = 10.0, dpi: int = 200,
                   title: str = "", zero: str = "F") -> list[Path]:
    if bands_result is None or dos_result is None:
        return []
    arrays = band_arrays(bands_result, fermi)
    data = dos_result.data.get("dos")
    if not data:
        return []
    eig, dist = arrays["eigenvalues"], arrays["distances"]
    ef = arrays["fermi"] or 0.0
    ticks, names = _tick_positions(dist, arrays["labels"])
    breaks = _path_breaks(dist, arrays.get("kpoints"))
    xdist = _broken(dist, breaks)
    energy = np.asarray(data["energy"]) - ef
    total = np.asarray(data.get("dos"))
    mask = (energy >= emin) & (energy <= emax)
    outdir = Path(outdir)
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, (ax, axd) = plt.subplots(
                1, 2, figsize=(8.8, 4.8), sharey=True, layout="constrained",
                gridspec_kw={"width_ratios": [2.6, 1.0]})
            fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
            colors = ["#1f4e9c", "#c0392b"]
            for spin in range(eig.shape[0]):
                for band in range(eig.shape[2]):
                    ax.plot(xdist, _broken(eig[spin, :, band] - ef, breaks),
                            lw=1.1, color=colors[spin % 2])
            ax.axhline(0.0, color="0.4", ls="--", lw=0.9)
            for tick in ticks:
                ax.axvline(tick, color="0.75", lw=0.7)
            ax.set_xticks(ticks)
            ax.set_xticklabels(names)
            ax.set_xlim(dist.min(), dist.max())
            ax.set_ylim(emin, emax)
            ax.set_ylabel(energy_axis_label(zero))

            axd.fill_betweenx(energy[mask], total[mask], color="0.85")
            axd.plot(total[mask], energy[mask], color="0.35", lw=1.2)
            channels = _pdos_channels(dos_result.data.get("pdos"), "per_element",
                                      energy + ef)
            for key, values in channels:
                axd.plot(values[mask], energy[mask], lw=1.1, label=key)
            axd.axhline(0.0, color="0.4", ls="--", lw=0.9)
            axd.set_xlabel("DOS")
            axd.set_xlim(left=0)
            if channels:
                axd.legend(frameon=False, fontsize=8)
            if title:
                fig.suptitle(title)
            path = outdir / "bands_dos.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                column_widths=[0.72, 0.28], horizontal_spacing=0.02)
            colors = ["#1f4e9c", "#c0392b"]
            for spin in range(eig.shape[0]):
                for band in range(eig.shape[2]):
                    fig.add_trace(go.Scatter(x=xdist,
                                             y=_broken(eig[spin, :, band] - ef, breaks),
                                             mode="lines", showlegend=False,
                                             line=dict(width=1.2, color=colors[spin % 2])),
                                  row=1, col=1)
            fig.add_trace(go.Scatter(x=total[mask], y=energy[mask], name="total DOS",
                                     fill="tozerox", line=dict(color="#444444")),
                          row=1, col=2)
            fig.update_yaxes(range=[emin, emax],
                             title_text=energy_axis_label(zero, latex=False),
                             row=1, col=1)
            fig.update_xaxes(tickvals=ticks, ticktext=[_plain(n) for n in names], row=1, col=1)
            fig.update_layout(template="plotly_white", title=title or "bands and DOS",
                              width=960, height=540)
            path = outdir / "bands_dos.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written


# ---------------------------------------------------------------- 収束履歴
def plot_convergence(results: Mapping[str, Any], outdir: str | Path,
                     backends: Iterable[str] = ("matplotlib",), dpi: int = 200) -> list[Path]:
    """各ステップの SCF / 構造最適化の履歴に沿った全エネルギーを描く。"""
    series = {name: res.data.get("scf_energies")
              for name, res in results.items()
              if getattr(res, "data", None) and res.data.get("scf_energies")}
    series = {k: v for k, v in series.items() if v and len(v) > 1}
    if not series:
        return []
    outdir = Path(outdir)
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, ax = plt.subplots(figsize=(6.0, 4.0))
            for name, values in series.items():
                values = np.asarray(values)
                ax.plot(np.arange(1, len(values) + 1), values - values[-1], "o-",
                        ms=3.5, lw=1.2, label=name)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_xlabel("SCF / ionic step")
            ax.set_ylabel(r"$E - E_\mathrm{final}$  (eV)")
            ax.legend(frameon=False, fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            path = outdir / "convergence.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)
        elif backend == "plotly":
            import plotly.graph_objects as go

            fig = go.Figure()
            for name, values in series.items():
                values = np.asarray(values)
                fig.add_trace(go.Scatter(x=list(range(1, len(values) + 1)),
                                         y=(values - values[-1]).tolist(),
                                         mode="lines+markers", name=name))
            fig.update_layout(template="plotly_white", title="convergence",
                              xaxis_title="step", yaxis_title="E - E_final (eV)",
                              width=760, height=460)
            path = outdir / "convergence.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written


# ------------------------------------------------------------ データ書き出し
def export_bands_csv(bands_result, path: str | Path, fermi: float | None = None) -> Path:
    arrays = band_arrays(bands_result, fermi)
    eig, dist, ef = arrays["eigenvalues"], arrays["distances"], arrays["fermi"]
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        header = ["k_distance"] + [f"spin{s}_band{b}"
                                   for s in range(eig.shape[0])
                                   for b in range(eig.shape[2])]
        fh.write(",".join(header) + "\n")
        for i, d in enumerate(dist):
            row = [f"{d:.6f}"] + [f"{eig[s, i, b] - ef:.6f}"
                                  for s in range(eig.shape[0])
                                  for b in range(eig.shape[2])]
            fh.write(",".join(row) + "\n")
    return path


def export_dos_csv(dos_result, path: str | Path, fermi: float | None = None) -> Path | None:
    data = dos_result.data.get("dos")
    if not data:
        return None
    ef = fermi if fermi is not None else (data.get("fermi_energy") or 0.0)
    energy = np.asarray(data["energy"]) - ef
    columns = {"E-EF": energy}
    for key in ("dos", "dos_up", "dos_down", "idos"):
        if data.get(key) is not None:
            columns[key] = np.asarray(data[key])
    for key, values in _pdos_channels(dos_result.data.get("pdos"), "per_orbital",
                                      energy + ef):
        columns[f"pdos_{key}"] = values
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for i in range(len(energy)):
            fh.write(",".join(f"{columns[k][i]:.6f}" for k in columns) + "\n")
    return path


# ------------------------------------------------------------------ 電荷密度
_AXIS_NAME = ("a", "b", "c")


def _charge_style(kind: str) -> tuple[str, str, bool]:
    """(カラーマップ, 物理量のラベル, 符号付きかどうか) を返す。"""
    if kind == "spin":
        return "RdBu_r", r"$\rho_\uparrow-\rho_\downarrow$  (e/bohr$^3$)", True
    if kind == "potential":
        return "viridis", "potential (Ry)", True
    return "magma", r"$\rho$  (e/bohr$^3$)", False


def plot_charge_profile(cube, outdir: str | Path, backends: Iterable[str] = ("matplotlib",),
                        kind: str = "density", dpi: int = 200,
                        title: str = "") -> list[Path]:
    """セルの各軸に沿った電荷密度の面平均を描く。"""
    from ezcal.charge import planar_average

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _, label, signed = _charge_style(kind)
    profiles = [planar_average(cube, axis) for axis in range(3)]
    symbols = cube.symbols
    fractional = cube.fractional_positions()
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), layout="constrained")
            for axis, (coords, values) in enumerate(profiles):
                panel = axes[axis]
                panel.plot(coords, values, color="#1f4e9c", lw=1.4)
                if signed:
                    panel.axhline(0.0, color="0.6", lw=0.8)
                length = coords[-1] + (coords[1] - coords[0] if len(coords) > 1 else 0)
                for index, symbol in enumerate(symbols):
                    panel.axvline(fractional[index, axis] % 1.0 * length,
                                  color="#b0392c", lw=0.7, alpha=0.55)
                panel.set_xlabel(f"position along {_AXIS_NAME[axis]}  (Å)")
                panel.set_xlim(coords.min(), coords.max())
                panel.grid(alpha=0.25)
            axes[0].set_ylabel(label)
            fig.suptitle(title or "planar average", fontsize=11)
            path = outdir / f"{kind}_profile.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=3,
                                subplot_titles=[f"along {n}" for n in _AXIS_NAME])
            for axis, (coords, values) in enumerate(profiles):
                fig.add_trace(go.Scatter(x=coords, y=values, mode="lines",
                                         line=dict(color="#1f4e9c"), showlegend=False),
                              row=1, col=axis + 1)
                fig.update_xaxes(title_text="Å", row=1, col=axis + 1)
            fig.update_layout(template="plotly_white", height=340, width=980,
                              title=title or "planar average")
            path = outdir / f"{kind}_profile.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written


def plot_charge_slice(cube, outdir: str | Path, backends: Iterable[str] = ("matplotlib",),
                      kind: str = "density", axis: int | None = None, fraction: float = 0.5,
                      dpi: int = 200, title: str = "") -> list[Path]:
    """電荷密度の 2 次元断面。

    ``axis=None`` (既定) ではセルの各軸につき 1 枚ずつ描く。自動で眺めるには
    こちらが便利。特定の面だけが欲しい場合は軸を指定する。
    """
    from ezcal.charge import slice_plane

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmap, label, signed = _charge_style(kind)
    axes = [axis] if axis is not None else [0, 1, 2]
    panels = [(a, *slice_plane(cube, axis=a, fraction=fraction)) for a in axes]
    limit = max(float(np.abs(data).max()) for _, data, _ in panels) or 1.0
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, grid = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0),
                                     layout="constrained", squeeze=False)
            image = None
            for panel, (a, data, extent) in zip(grid[0], panels):
                others = [i for i in range(3) if i != a]
                kwargs = {"cmap": cmap, "origin": "lower", "extent": extent,
                          "aspect": "auto"}
                if signed:
                    kwargs.update(vmin=-limit, vmax=limit)
                image = panel.imshow(data, **kwargs)
                panel.set_xlabel(f"{_AXIS_NAME[others[1]]}  (Å)")
                panel.set_ylabel(f"{_AXIS_NAME[others[0]]}  (Å)")
                panel.set_title(f"{_AXIS_NAME[a]} = {fraction:.2f}", fontsize=10)
            fig.colorbar(image, ax=grid[0].tolist(), label=label, shrink=0.85)
            fig.suptitle(title or kind, fontsize=11)
            path = outdir / f"{kind}_slice.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=1, cols=len(panels),
                                subplot_titles=[f"{_AXIS_NAME[a]} = {fraction:.2f}"
                                                for a, _, _ in panels])
            for index, (a, data, _) in enumerate(panels):
                fig.add_trace(go.Heatmap(
                    z=data, colorscale="RdBu_r" if signed else "Magma",
                    zmid=0 if signed else None, zmin=-limit if signed else None,
                    zmax=limit if signed else None, showscale=index == 0,
                    colorbar=dict(title=_plain(label))), row=1, col=index + 1)
            fig.update_layout(template="plotly_white", width=380 * len(panels), height=420,
                              title=title or kind)
            path = outdir / f"{kind}_slice.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written




# ------------------------------------------------------- 電荷密度の 3D 表示
#: 3D 表示で使う原子の色 (CPK 風。ここに無い元素は既定色)
_ATOM_COLORS = {
    "H": "#ffffff", "Li": "#cc80ff", "Be": "#c2ff00", "B": "#ffb5b5", "C": "#404040",
    "N": "#3050f8", "O": "#ff0d0d", "F": "#90e050", "Na": "#ab5cf2", "Mg": "#8aff00",
    "Al": "#bfa6a6", "Si": "#f0c8a0", "P": "#ff8000", "S": "#ffff30", "Cl": "#1ff01f",
    "K": "#8f40d4", "Ca": "#3dff00", "Ti": "#bfc2c7", "Cr": "#8a99c7", "Mn": "#9c7ac7",
    "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050", "Cu": "#c88033", "Zn": "#7d80b0",
    "Ga": "#c28f8f", "Ge": "#668f8f", "Zr": "#94e0e0", "Nb": "#73c2c9", "Mo": "#54b5b5",
    "Ag": "#c0c0c0", "Sn": "#668080", "Ba": "#00c900", "W": "#2194d6", "Pt": "#d0d0e0",
    "Au": "#ffd123", "Pb": "#575961",
}
_DEFAULT_ATOM_COLOR = "#909090"


def _colormap(name: str):
    """matplotlib のバージョン差 (cm.get_cmap の廃止) を吸収する。"""
    import matplotlib

    try:
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):               # pragma: no cover - 古い matplotlib
        from matplotlib import cm

        return cm.get_cmap(name)


def _cell_edges(lattice, origin=None):
    """セルの 12 稜線を (始点, 終点) の組で返す。"""
    lattice = np.asarray(lattice, dtype=float)
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    corners = {}
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                corners[(i, j, k)] = origin + np.array([i, j, k]) @ lattice
    edges = []
    for corner in corners:
        for axis in range(3):
            neighbour = list(corner)
            if neighbour[axis]:
                continue
            neighbour[axis] = 1
            edges.append((corners[corner], corners[tuple(neighbour)]))
    return edges


def _iso_levels(data: np.ndarray, kind: str, levels: Sequence[float] | None = None):
    """描く等値面の値。指定が無ければ密度分布の分位点から決める。"""
    if levels:
        return [float(v) for v in levels], kind in {"spin", "potential"}
    signed = kind in {"spin", "potential"}
    if signed:
        limit = float(np.abs(data).max()) or 1.0
        return [-0.35 * limit, 0.35 * limit], True
    finite = data[data > 0]
    if not finite.size:
        return [float(data.max())], False
    return [float(np.percentile(finite, 92)), float(np.percentile(finite, 99))], False


def _grid_points(cube, data: np.ndarray) -> np.ndarray:
    n1, n2, n3 = data.shape
    grid = np.stack(np.meshgrid(np.linspace(0, 1, n1, endpoint=False),
                                np.linspace(0, 1, n2, endpoint=False),
                                np.linspace(0, 1, n3, endpoint=False),
                                indexing="ij"), axis=-1)
    return grid.reshape(-1, 3) @ np.asarray(cube.lattice, dtype=float)


def _downsample(data: np.ndarray, max_points: int) -> np.ndarray:
    stride = [max(1, int(np.ceil(n / max_points))) for n in data.shape]
    return data[::stride[0], ::stride[1], ::stride[2]]


def _marching_cubes(data: np.ndarray, level: float, lattice):
    """周期境界で閉じた等値面 (頂点 Angstrom, 面) を返す。skimage が無ければ None。"""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        return None
    padded = np.pad(data, 1, mode="wrap")            # 端でも面が閉じるようにする
    if not (padded.min() < level < padded.max()):
        return None
    verts, faces, _, _ = marching_cubes(padded, level=level)
    fractional = (verts - 1.0) / np.array(data.shape, dtype=float)
    return fractional @ np.asarray(lattice, dtype=float), faces


def plot_charge_isosurface(cube, outdir: str | Path, kind: str = "density",
                           max_points: int = 64, title: str = "",
                           backends: Iterable[str] = ("plotly",),
                           levels: Sequence[float] | None = None,
                           dpi: int = 200, opacity: float = 0.45) -> list[Path]:
    """電荷密度を空間にマッピングした 3D 等値面。

    plotly 版は回転・拡大ができる対話的な図 (html)、matplotlib 版は報告書に
    貼れる静止画 (png) を書き出す。静止画では marching cubes で三角形分割した
    等値面を描き、``scikit-image`` が無い環境では密度の高いボクセルを点群と
    して描くことで代用する。原子とセルの稜線はどちらの版にも重ねて描く。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    backends = list(backends) or ["plotly"]
    data = _downsample(np.asarray(cube.density, dtype=float), max_points)
    values, signed = _iso_levels(data, kind, levels)
    colorscale = "RdBu_r" if signed else "Magma"
    symbols = cube.symbols
    positions = (np.asarray(cube.positions, dtype=float)
                 if cube.positions is not None else np.zeros((0, 3)))
    written: list[Path] = []

    for backend in backends:
        if backend == "plotly":
            import plotly.graph_objects as go

            cartesian = _grid_points(cube, data)
            fig = go.Figure(go.Isosurface(
                x=cartesian[:, 0], y=cartesian[:, 1], z=cartesian[:, 2],
                value=data.ravel(), isomin=min(values), isomax=max(values),
                surface_count=max(2, len(values)), opacity=opacity,
                colorscale=colorscale,
                caps=dict(x_show=False, y_show=False, z_show=False),
                colorbar=dict(title="e/bohr³"), name=kind))
            if len(positions):
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                    mode="markers+text", text=symbols, textposition="top center",
                    marker=dict(size=7, color=[_ATOM_COLORS.get(s, _DEFAULT_ATOM_COLOR)
                                               for s in symbols],
                                line=dict(color="#333333", width=1)),
                    name="atoms"))
            for start, end in _cell_edges(cube.lattice, cube.origin):
                fig.add_trace(go.Scatter3d(
                    x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                    mode="lines", line=dict(color="#8a8a8a", width=2),
                    showlegend=False, hoverinfo="skip"))
            fig.update_layout(template="plotly_white", width=780, height=700,
                              title=title or f"{kind} isosurface",
                              scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)",
                                         zaxis_title="z (Å)", aspectmode="data"))
            path = outdir / f"{kind}_isosurface.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)

        elif backend == "matplotlib":
            plt = _mpl()
            from matplotlib import colors as mcolors
            from matplotlib.cm import ScalarMappable
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(7.2, 6.4))
            panel = fig.add_subplot(111, projection="3d")
            cmap = _colormap("RdBu_r" if signed else "magma")
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            ordered = sorted(values, key=abs)
            span = max(1, len(ordered) - 1)
            handles, drawn = [], False
            for index, level in enumerate(ordered):
                # 外側 (低い等値面) ほど薄く描かないと内側が見えない
                colour = (cmap(norm(level)) if signed
                          else cmap(0.45 + 0.45 * index / span))
                alpha = opacity * (0.55 + 0.45 * index / span)
                mesh = _marching_cubes(data, level, cube.lattice)
                if mesh is None:
                    continue
                verts, faces = mesh
                collection = Poly3DCollection(verts[faces], alpha=alpha,
                                              linewidths=0.0)
                collection.set_facecolor(colour)
                panel.add_collection3d(collection)
                handles.append(plt.Line2D([], [], marker="s", linestyle="",
                                          markersize=9, color=colour,
                                          label=f"{level:.3g} e/bohr³"))
                drawn = True
            if not drawn:                             # skimage が無いときの代替表示
                cloud = _grid_points(cube, data)
                flat = data.ravel()
                keep = np.abs(flat) >= min(abs(v) for v in values)
                panel.scatter(cloud[keep, 0], cloud[keep, 1], cloud[keep, 2],
                              c=flat[keep], cmap=cmap, s=4, alpha=0.25,
                              linewidths=0)
            for start, end in _cell_edges(cube.lattice, cube.origin):
                panel.plot(*zip(start, end), color="#8a8a8a", lw=0.8)
            if len(positions):
                panel.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                              c=[_ATOM_COLORS.get(s, _DEFAULT_ATOM_COLOR)
                                 for s in symbols],
                              s=90, edgecolors="#333333", linewidths=0.6, depthshade=False)
                for symbol, position in zip(symbols, positions):
                    panel.text(*position, f" {symbol}", fontsize=8)
            if handles:
                panel.legend(handles=handles, loc="upper left", frameon=False,
                             fontsize=9, title="isosurface level")
            else:
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=panel,
                             shrink=0.65, pad=0.08, label="ρ (e/bohr³)")
            panel.set_xlabel("x (Å)")
            panel.set_ylabel("y (Å)")
            panel.set_zlabel("z (Å)")
            panel.set_title(title or f"{kind} isosurface", fontsize=11)
            _equal_aspect_3d(panel, cube.lattice, cube.origin)
            path = outdir / f"{kind}_isosurface.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


def _equal_aspect_3d(panel, lattice, origin=None) -> None:
    """3D 軸の縦横比をセルの実寸に合わせる。"""
    corners = np.array([start for start, _ in _cell_edges(lattice, origin)])
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    centre = (lower + upper) / 2.0
    radius = float(np.max(upper - lower)) / 2.0 or 1.0
    panel.set_xlim(centre[0] - radius, centre[0] + radius)
    panel.set_ylim(centre[1] - radius, centre[1] + radius)
    panel.set_zlim(centre[2] - radius, centre[2] + radius)
    try:
        panel.set_box_aspect((1, 1, 1))
    except Exception:                                 # pragma: no cover - 古い matplotlib
        pass


# --------------------------------------------------- 原子電荷の 3D マッピング
#: 原子電荷として使える列 (優先順)
CHARGE_COLUMNS = (
    ("bader_charge", "Bader charge (e)"),
    ("lowdin_charge", "Löwdin charge (e)"),
    ("bader_electrons", "Bader electrons (e)"),
    ("lowdin_electrons", "Löwdin electrons (e)"),
    ("moment_sphere", "magnetic moment (μB)"),
    ("lowdin_moment", "Löwdin moment (μB)"),
)


def charge_column(rows: Sequence[Mapping], source: str = "auto") -> tuple[str, str]:
    """3D マッピングに使う列を決める。``(列名, ラベル)``。"""
    available = {key for row in rows for key, value in row.items() if value is not None}
    if source and source != "auto":
        for key, label in CHARGE_COLUMNS:
            if key == source:
                return key, label
        return source, source
    for key, label in CHARGE_COLUMNS:
        if key in available:
            return key, label
    raise ValueError("原子電荷の列が見つかりません (先に charge タスクを実行してください)")


def plot_charge_map_3d(rows: Sequence[Mapping], outdir: str | Path,
                       backends: Iterable[str] = ("matplotlib",),
                       lattice=None, source: str = "auto", dpi: int = 200,
                       title: str = "", name: str = "charge_map") -> list[Path]:
    """原子ごとの価数を 3D 空間にマッピングする。

    各原子を実座標に配置し、価数 (既定では Bader 電荷、無ければ Löwdin 電荷) で
    色と大きさを変え、値を文字で添える。陽イオン (電子を失った側) が赤、陰イオンが
    青になるよう 0 を中心とした発散カラーマップを使う。``rows`` は
    :func:`ezcal.charge.atomic_charge_table` が返す行 (= ``atomic_charges.csv``)。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in rows if row.get("x") is not None]
    if not rows:
        return []
    key, label = charge_column(rows, source)
    values = np.array([float(row.get(key) or 0.0) for row in rows])
    positions = np.array([[float(row["x"]), float(row["y"]), float(row["z"])]
                          for row in rows])
    symbols = [str(row.get("element") or row.get("label") or "?") for row in rows]
    labels = [str(row.get("label") or row.get("element") or "?") for row in rows]
    limit = float(np.abs(values).max()) or 1.0
    sizes = 120.0 + 340.0 * np.abs(values) / limit
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            from matplotlib import colors as mcolors
            from matplotlib.cm import ScalarMappable

            fig = plt.figure(figsize=(7.4, 6.4))
            panel = fig.add_subplot(111, projection="3d")
            norm = mcolors.Normalize(vmin=-limit, vmax=limit)
            cmap = _colormap("RdBu_r")
            if lattice is not None:
                for start, end in _cell_edges(lattice):
                    panel.plot(*zip(start, end), color="#8a8a8a", lw=0.8)
            panel.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                          c=values, cmap=cmap, norm=norm, s=sizes,
                          edgecolors="#222222", linewidths=0.7, depthshade=False)
            for position, symbol, value in zip(positions, labels, values):
                panel.text(*position, f"  {symbol} {value:+.2f}", fontsize=8)
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=panel,
                         shrink=0.65, pad=0.08, label=label)
            panel.set_xlabel("x (Å)")
            panel.set_ylabel("y (Å)")
            panel.set_zlabel("z (Å)")
            panel.set_title(title or f"atomic charges — {label}", fontsize=11)
            if lattice is not None:
                _equal_aspect_3d(panel, lattice)
            path = outdir / f"{name}.png"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go

            hover = []
            for row, value in zip(rows, values):
                parts = [f"<b>{row.get('label') or row.get('element')}</b> "
                         f"(#{row.get('index')})", f"{label}: {value:+.3f}"]
                for extra_key, extra_label in CHARGE_COLUMNS:
                    if extra_key != key and row.get(extra_key) is not None:
                        parts.append(f"{extra_label}: {float(row[extra_key]):+.3f}")
                if row.get("bader_volume") is not None:
                    parts.append(f"Bader volume: {float(row['bader_volume']):.2f} Å³")
                hover.append("<br>".join(parts))
            fig = go.Figure(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode="markers+text",
                text=[f"{s} {v:+.2f}" for s, v in zip(symbols, values)],
                textposition="top center", hovertext=hover, hoverinfo="text",
                marker=dict(size=10.0 + 16.0 * np.abs(values) / limit,
                            color=values, colorscale="RdBu_r", cmin=-limit, cmax=limit,
                            colorbar=dict(title=_plain(label)),
                            line=dict(color="#222222", width=1)),
                name="atoms"))
            if lattice is not None:
                for start, end in _cell_edges(lattice):
                    fig.add_trace(go.Scatter3d(
                        x=[start[0], end[0]], y=[start[1], end[1]],
                        z=[start[2], end[2]], mode="lines",
                        line=dict(color="#8a8a8a", width=2),
                        showlegend=False, hoverinfo="skip"))
            fig.update_layout(template="plotly_white", width=780, height=700,
                              title=title or f"atomic charges — {_plain(label)}",
                              scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)",
                                         zaxis_title="z (Å)", aspectmode="data"))
            path = outdir / f"{name}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written


# ------------------------------------------------------------ MD / MC の記録
#: energy_log.csv の列 -> (縦軸ラベル, パネルの見出し)
DYNAMICS_PANELS = (
    ("energy_eV", "E$_{pot}$ (eV)", "potential energy"),
    ("temperature_K", "T (K)", "temperature"),
    ("e_total_eV", "E$_{tot}$ (eV)", "total energy"),
    ("volume_A3", "V (Å$^3$)", "volume"),
    ("msd_A2", "MSD (Å$^2$)", "mean square displacement"),
)

_PHASE_COLORS = ("#1f4e9c", "#b0392c", "#2f8f52", "#8a5cc7", "#c98a1e", "#3aa3b5")


def _phase_segments(phases: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    """フェーズが連続している区間ごとに分ける。

    MCMD のようにフェーズが交互に現れる記録では、同じフェーズの点を 1 本の線で
    結ぶと、間に挟まったブロックの上を線が渡ってしまい「その間も値が続いていた」
    ように見えてしまう。区間で切っておけば、記録が無いところは線が引かれない。
    """
    segments: list[tuple[str, np.ndarray]] = []
    start = 0
    for index in range(1, len(phases) + 1):
        if index == len(phases) or phases[index] != phases[start]:
            segments.append((phases[start], np.arange(start, index)))
            start = index
    return segments


def plot_dynamics(records: Sequence[Mapping], outdir: str | Path,
                  backends: Iterable[str] = ("matplotlib",), dpi: int = 200,
                  title: str = "", name: str = "dynamics") -> list[Path]:
    """MD / MC の記録 (``energy_log.csv``) を時系列で描く。

    エネルギー・温度・全エネルギー・体積・MSD のうち、記録に存在する量だけを
    縦に並べる。フェーズ (mcmc / md-nvt / relax ...) は色で区別する。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    if not records:
        return []
    steps = np.array([float(row.get("step", index))
                      for index, row in enumerate(records)])
    phases = [str(row.get("phase", "")) for row in records]
    order = list(dict.fromkeys(phases))
    colors = {phase: _PHASE_COLORS[index % len(_PHASE_COLORS)]
              for index, phase in enumerate(order)}

    segments = _phase_segments(phases)
    panels = []
    for key, ylabel, heading in DYNAMICS_PANELS:
        values = np.array([float(row[key]) if row.get(key) is not None else np.nan
                           for row in records])
        if np.isfinite(values).sum() > 1:
            panels.append((key, ylabel, heading, values))
    if not panels:
        return []
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, grid = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                                     figsize=(8.4, 2.3 * len(panels) + 0.8),
                                     layout="constrained")
            for (key, ylabel, heading, values), panel in zip(panels, grid[:, 0]):
                seen: set[str] = set()
                for phase, index in segments:
                    if not np.isfinite(values[index]).any():
                        continue
                    label = None
                    if key == panels[0][0] and phase not in seen:
                        label = phase
                        seen.add(phase)
                    panel.plot(steps[index], values[index], ".-", ms=2.2, lw=0.9,
                               color=colors[phase], label=label)
                panel.set_ylabel(ylabel)
                panel.grid(alpha=0.25)
                panel.set_title(heading, fontsize=9, loc="left", color="0.35")
            grid[-1, 0].set_xlabel("global step")
            if len(order) > 1:
                grid[0, 0].legend(fontsize=8, ncols=min(4, len(order)), frameon=False)
            fig.suptitle(title or "dynamics", fontsize=11)
            path = outdir / f"{name}.png"
            fig.savefig(path, dpi=dpi)
            plt.close(fig)
            written.append(path)

        elif backend == "plotly":
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                                subplot_titles=[heading for _, _, heading, _ in panels],
                                vertical_spacing=0.06)
            for row_index, (key, ylabel, _, values) in enumerate(panels, start=1):
                seen = set()
                for phase, index in segments:
                    if not np.isfinite(values[index]).any():
                        continue
                    show = row_index == 1 and phase not in seen
                    seen.add(phase)
                    fig.add_trace(go.Scatter(
                        x=steps[index], y=values[index], mode="lines+markers",
                        marker=dict(size=3), line=dict(color=colors[phase], width=1.2),
                        name=phase, legendgroup=phase,
                        showlegend=show), row=row_index, col=1)
                fig.update_yaxes(title_text=_plain(ylabel), row=row_index, col=1)
            fig.update_xaxes(title_text="global step", row=len(panels), col=1)
            fig.update_layout(template="plotly_white", height=260 * len(panels) + 120,
                              width=960, title=title or "dynamics")
            path = outdir / f"{name}.html"
            fig.write_html(path, include_plotlyjs="cdn")
            written.append(path)
    return written
