"""Plotting of band structures, densities of states and relaxation traces.

Every function takes a ``backends`` list - any of ``matplotlib``,
``plotly``, ``both`` or ``none`` - and returns the files it wrote.
"""

from __future__ import annotations

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
    """Put a PDOS channel on the DOS energy grid (the two tools use different grids)."""
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
    """One colour per orbital, shared by its up and down halves."""
    colours, order = {}, []
    for name, _ in channels:
        base = name.rsplit(" ", 1)[0] if name.endswith((" up", " down")) else name
        if base not in order:
            order.append(base)
        colours[name] = _PALETTE[order.index(base) % len(_PALETTE)]
    return colours


def energy_axis_label(zero: str = "F", latex: bool = True) -> str:
    """Axis label naming what the energy zero is (E_F for metals, VBM otherwise)."""
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


# --------------------------------------------------------------- band data
def band_arrays(bands_result, fermi: float | None = None) -> dict:
    """Normalise a bands :class:`CalcResult` into plain arrays."""
    data = bands_result.data
    eig = np.asarray(data["eigenvalues"])                # (nspin, nk, nbnd)
    kpath = data.get("kpath", {})
    distances = np.asarray(kpath.get("distances", np.arange(eig.shape[1])), dtype=float)
    labels = [(int(i), str(lab)) for i, lab in kpath.get("labels", [])]
    ef = fermi if fermi is not None else (bands_result.fermi_energy or 0.0)
    return {"eigenvalues": eig, "distances": distances, "labels": labels, "fermi": ef}


def _tick_positions(distances: np.ndarray, labels: Sequence[tuple[int, str]]):
    """Group labels that sit at the same path length (band path breaks)."""
    grouped: dict[int, tuple[float, str]] = {}
    for index, name in labels:
        if not (0 <= index < len(distances)) or not name:
            continue
        position = float(distances[index])
        # group by path length, but keep the exact value: a rounded tick can
        # fall marginally outside the axis limits and matplotlib drops it
        key = int(round(position * 1e6))
        if key not in grouped:
            grouped[key] = (position, name)
        else:
            _, previous = grouped[key]
            if name not in previous.split("|"):
                grouped[key] = (position, f"{previous}|{name}")
    ordered = [grouped[k] for k in sorted(grouped)]
    return [pos for pos, _ in ordered], [name for _, name in ordered]


# ------------------------------------------------------------------- bands
def plot_bands(bands_result, outdir: str | Path, backends: Iterable[str] = ("matplotlib",),
               fermi: float | None = None, emin: float | None = None,
               emax: float | None = None, dpi: int = 200,
               title: str = "band structure", zero: str = "F") -> list[Path]:
    arrays = band_arrays(bands_result, fermi)
    eig, dist = arrays["eigenvalues"], arrays["distances"]
    ef = arrays["fermi"]
    ticks, names = _tick_positions(dist, arrays["labels"])
    lo = emin if emin is not None else -10.0
    hi = emax if emax is not None else 10.0
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for backend in backends:
        if backend == "matplotlib":
            plt = _mpl()
            fig, ax = plt.subplots(figsize=(6.0, 4.6))
            colors = ["#1f4e9c", "#c0392b"]
            for spin in range(eig.shape[0]):
                for band in range(eig.shape[2]):
                    ax.plot(dist, eig[spin, :, band] - ef, lw=1.1,
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
                        x=dist, y=eig[spin, :, band] - ef, mode="lines",
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


# --------------------------------------------------------------------- dos
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


# ------------------------------------------------------------ bands + dos
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
                    ax.plot(dist, eig[spin, :, band] - ef, lw=1.1, color=colors[spin % 2])
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
                    fig.add_trace(go.Scatter(x=dist, y=eig[spin, :, band] - ef,
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


# ------------------------------------------------------------ convergence
def plot_convergence(results: Mapping[str, Any], outdir: str | Path,
                     backends: Iterable[str] = ("matplotlib",), dpi: int = 200) -> list[Path]:
    """Total energy along the SCF / relaxation history of every step."""
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


# ------------------------------------------------------------- data export
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


# ------------------------------------------------------------ charge density
_AXIS_NAME = ("a", "b", "c")


def _charge_style(kind: str) -> tuple[str, str, bool]:
    """(colour map, quantity label, is the quantity signed?)"""
    if kind == "spin":
        return "RdBu_r", r"$\rho_\uparrow-\rho_\downarrow$  (e/bohr$^3$)", True
    if kind == "potential":
        return "viridis", "potential (Ry)", True
    return "magma", r"$\rho$  (e/bohr$^3$)", False


def plot_charge_profile(cube, outdir: str | Path, backends: Iterable[str] = ("matplotlib",),
                        kind: str = "density", dpi: int = 200,
                        title: str = "") -> list[Path]:
    """Planar average of the density along each cell axis."""
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
    """2D cuts through the density.

    With ``axis=None`` (the default) one panel per cell axis is drawn, which
    is the more useful thing to look at automatically; give an axis to get a
    single specific plane.
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


def plot_charge_isosurface(cube, outdir: str | Path, kind: str = "density",
                           max_points: int = 64, title: str = "") -> list[Path]:
    """Interactive 3D isosurface (plotly only - a static one adds nothing)."""
    import plotly.graph_objects as go

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = np.asarray(cube.density, dtype=float)
    stride = [max(1, int(np.ceil(n / max_points))) for n in data.shape]
    data = data[::stride[0], ::stride[1], ::stride[2]]

    n1, n2, n3 = data.shape
    grid = np.stack(np.meshgrid(np.linspace(0, 1, n1, endpoint=False),
                                np.linspace(0, 1, n2, endpoint=False),
                                np.linspace(0, 1, n3, endpoint=False),
                                indexing="ij"), axis=-1)
    cartesian = grid.reshape(-1, 3) @ cube.lattice

    signed = kind in {"spin", "potential"}
    if signed:
        limit = float(np.abs(data).max())
        levels = (-0.25 * limit, 0.25 * limit)
        colorscale = "RdBu_r"
    else:
        finite = data[data > 0]
        level = float(np.percentile(finite, 96)) if finite.size else float(data.max())
        levels = (level, float(data.max()))
        colorscale = "Magma"

    fig = go.Figure(go.Isosurface(
        x=cartesian[:, 0], y=cartesian[:, 1], z=cartesian[:, 2],
        value=data.ravel(), isomin=levels[0], isomax=levels[1],
        surface_count=3, opacity=0.45, colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="e/bohr³")))
    if cube.positions is not None and len(cube.positions):
        fig.add_trace(go.Scatter3d(
            x=cube.positions[:, 0], y=cube.positions[:, 1], z=cube.positions[:, 2],
            mode="markers+text", text=cube.symbols, textposition="top center",
            marker=dict(size=6, color="#1f4e9c"), name="atoms"))
    fig.update_layout(template="plotly_white", width=760, height=680,
                      title=title or f"{kind} isosurface",
                      scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)",
                                 zaxis_title="z (Å)", aspectmode="data"))
    path = outdir / f"{kind}_isosurface.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return [path]
