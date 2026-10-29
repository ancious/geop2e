#!/usr/bin/env python3
"""Render the supplied Fig. 12 data using Matplotlib only.

Only this script and fig8_gpt6.png are project outputs. The JSON is read-only.
Data take precedence over the prose in the supplied draft caption:
* context.cert_meaning identifies cert as raw lambda_min, not log10(lambda_min).
* the ROC is recomputed from failure rates, using the stated strict 50% rule.
* the two CAD pitches are reference locations, not asserted exact error modes.
* panels a/b describe the deployed profile; panel c describes the unbounded arm.

Exact coincident scatter coordinates are consolidated without jitter; marker
area counts all contributing trials. The broken logarithmic certificate axis
omits an empty interval only. No histogram samples are discarded or clipped.

Layout iteration after inspecting the first render: shorten the CAD reference
lines and attach leaders below their labels (the initial lines crossed a label),
mask the ROC annotation background, increase tick type, and identify the actual
most-repeated error rather than imply that a CAD pitch is an exact error mode.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from collections import Counter

# Matplotlib's cache is outside the project and is removed on exit.
_cache = tempfile.TemporaryDirectory(prefix="fig12-mpl-")
os.environ["MPLCONFIGDIR"] = _cache.name
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, NullLocator
from matplotlib.transforms import Bbox
import numpy as np


DATA = Path(__file__).resolve().parents[2] / "results" / "e9" / "fig12_data.json"
OUTPUT = Path(__file__).resolve().parent / "fig8_gpt6.png"
DPI = 300
WIDTH, HEIGHT = 6.5, 6.3
BLUE = "#2980b9"   # flat-UI blue, matches original submission figures (GeoP2E/success)
ORANGE = "#c0392b"  # flat-UI red, matches original submission figures (failure/warning)
INK = "#222222"
GRAY = "#656565"
LIGHT = "#F2F2F2"
MIN_TICK_SIZE = 8.5


def read_data():
    source_bytes = DATA.read_bytes()
    d = json.loads(source_bytes)
    ctx = d["context"]
    success_limit = float(re.search(r"success threshold ([\d.]+) mm", ctx["e_tte_meaning"])[1])
    tau = float(re.search(r"threshold ([\d.]+)", ctx["tau_context"])[1])
    absent_folds, total_folds = map(int, re.search(r"(\d+) of (\d+) folds", ctx["tau_context"]).groups())
    positive_rate = float(re.search(r"below ([\d.]+)%", d["roc"]["positive_class"])[1]) / 100
    trials, cells = d["scatter"], d["roc"]["cells"]
    assert len(trials) == ctx["n_trials"]
    assert all(r["cert"] > 0 and r["e_tte"] > 0 for r in trials)
    assert all(r["success"] == (r["e_tte"] < success_limit) for r in trials)
    cell_sizes = []
    for c in cells:
        members = [r for r in trials if (r["part"], r["cert"]) == (c["part"], c["cert"])]
        cell_sizes.append(len(members))
        assert np.isclose(np.mean([not r["success"] for r in members]), c["fail_rate"])
    assert len(set(cell_sizes)) == 1
    certificate = np.array([c["cert"] for c in cells])
    positive = np.array([(1 - c["fail_rate"]) <= positive_rate for c in cells])
    # Lower lambda_min is a stronger failure flag. Include every unique cutoff.
    cutoffs = np.r_[-np.inf, np.unique(certificate)]
    predicted = certificate[None, :] <= cutoffs[:, None]
    tpr = (predicted & positive).sum(axis=1) / positive.sum()
    fpr = (predicted & ~positive).sum(axis=1) / (~positive).sum()
    auc = float(np.trapezoid(tpr, fpr))
    pairwise_auc = np.mean(
        (certificate[positive, None] < certificate[None, ~positive])
        + 0.5 * (certificate[positive, None] == certificate[None, ~positive])
    )
    assert np.isclose(auc, pairwise_auc)
    flagged = certificate < tau
    op_tpr = float((flagged & positive).sum() / positive.sum())
    op_fpr = float((flagged & ~positive).sum() / (~positive).sum())
    # The large empty gap immediately below the populated finite range.
    logcert = np.sort(np.unique(np.log10(certificate)))
    gaps = np.diff(logcert)
    candidates = np.where((gaps > 5) & (logcert[1:] < 0))[0]
    split = int(candidates[-1])
    low_max = 10.0 ** math.ceil(logcert[split] + 0.5)
    high_min = 10.0 ** math.floor(logcert[split + 1])
    left_min = 10.0 ** math.floor(logcert[0] - 0.5)
    right_max = 10.0 ** (math.log10(max(certificate)) + 0.20)
    assert not np.any((certificate > low_max) & (certificate < high_min))
    hist = np.array(d["failure_hist"])
    assert np.all(hist > success_limit)
    return dict(
        d=d, ctx=ctx, trials=trials, cells=cells, hist=hist,
        n_trials=len(trials), n_cells=len(cells), n_parts=len({r["part"] for r in trials}),
        per_cell=cell_sizes[0], success_limit=success_limit, tau=tau,
        absent_folds=absent_folds, total_folds=total_folds, positive_rate=positive_rate,
        positive=positive, auc=auc, fpr=fpr, tpr=tpr, op_tpr=op_tpr, op_fpr=op_fpr,
        left_min=left_min, low_max=low_max, high_min=high_min, right_max=right_max,
        source_hash=hashlib.sha256(source_bytes).hexdigest(),
    )


def figure_text(fig, x, y, label, **kwargs):
    return fig.text(x, y, label, color=INK, ha="left", va="top", **kwargs)


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="major", width=0.65, length=3, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E4E4E4", linewidth=0.5)


def build_figure(s):
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8.5,
        "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": MIN_TICK_SIZE, "ytick.labelsize": MIN_TICK_SIZE,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.65, "mathtext.fontset": "stix",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI)
    figure_text(fig, 0.025, 0.985, "Single-view occlusion: observability and error structure",
                fontsize=11.2, fontweight="bold")
    figure_text(fig, 0.025, 0.948,
                f"{s['n_parts']} parts  ·  {s['n_cells']} (part, viewpoint) cells  ·  "
                f"{s['per_cell']} initializations per cell  ·  {s['n_trials']} trials", fontsize=8.5)
    figure_text(fig, 0.100, 0.906, "(a)  Offline certificate versus trial outcome", fontweight="bold", fontsize=9.5)
    figure_text(fig, 0.100, 0.878, s["ctx"]["arm"], fontsize=8.5)

    left = fig.add_axes([0.100, 0.545, 0.180, 0.290])
    main = fig.add_axes([0.315, 0.545, 0.660, 0.290], sharey=left)
    axes = [left, main]
    errors = np.array([r["e_tte"] for r in s["trials"]])
    ymin, ymax = 10 ** (np.log10(errors.min()) - 0.20), 10 ** math.ceil(np.log10(errors.max()))
    for ax in axes:
        clean_axis(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(FixedLocator([10.0 ** k for k in range(math.ceil(math.log10(ymin)), math.floor(math.log10(ymax)) + 1)]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axhline(s["success_limit"], color=INK, linewidth=0.8, linestyle=(0, (5, 3)))
    left.set_xlim(s["left_min"], s["low_max"])
    main.set_xlim(s["high_min"], s["right_max"])
    left.set_xticks([10.0 ** k for k in range(math.ceil(math.log10(s["left_min"])) + 1, math.floor(math.log10(s["low_max"])) + 1, 9)])
    main.set_xticks([10.0 ** k for k in range(math.ceil(math.log10(s["high_min"])), math.floor(math.log10(s["right_max"])) + 1)])
    for ax in axes:
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
    left.set_ylabel(r"Whole-model pose error $e_{\mathrm{tte}}$ (mm)", labelpad=6)
    main.tick_params(axis="y", left=False, labelleft=False)
    main.spines["left"].set_visible(False)
    left.axvspan(*left.get_xlim(), color=LIGHT, zorder=0)
    main.axvspan(main.get_xlim()[0], s["tau"], color=LIGHT, zorder=0)
    main.axvline(s["tau"], color=INK, linewidth=1.0, linestyle=(0, (3, 2)), zorder=2)
    main.text(s["tau"] * 0.84, ymax * 0.72, rf"$\tau = {s['tau']:g}$", ha="right", va="center", fontsize=8.5)
    # Break marks, in physical points, on the two interrupted horizontal edges.
    for ax, xpos in [(left, 1), (main, 0)]:
        ax.plot([xpos], [0], transform=ax.transAxes, marker=[(-1, -0.7), (1, 0.7)],
                markersize=7, linestyle="none", markeredgewidth=0.8, color=INK, clip_on=False)

    counts = Counter((r["cert"], r["e_tte"], r["success"]) for r in s["trials"])
    base_area = 8.0
    represented = 0
    for ax in axes:
        lo, hi = ax.get_xlim()
        for outcome, marker, color in [(False, "^", ORANGE), (True, "o", BLUE)]:
            points = [(key[0], key[1], n) for key, n in counts.items() if lo <= key[0] <= hi and key[2] == outcome]
            if points:
                x, y, n = np.array(points).T
                represented += int(n.sum())
                ax.scatter(x, y, s=base_area * n, marker=marker,
                           facecolors=color if not outcome else "white", edgecolors=color,
                           linewidths=0.7, alpha=0.92, zorder=4 if outcome else 3)
    assert represented == s["n_trials"]
    near = [r for r in s["trials"] if r["cert"] <= s["low_max"]]
    left.text(0.05, 0.95, f"Near zero\n{sum(not r['success'] for r in near)}/{len(near)} failures",
              transform=left.transAxes, va="top", fontsize=8.4, linespacing=1.4)
    main.text(0.025, 0.405, f"Success cutoff: {s['success_limit']:g} mm", transform=main.transAxes,
              fontsize=8.0, va="bottom", bbox=dict(facecolor=LIGHT, edgecolor="none", pad=0.8))
    fig.text(0.540, 0.493, r"Offline CAD certificate $\lambda_{\min}$ (log scale)", ha="center", va="top", fontsize=9)
    # Outcome encoding and count key use color and shape, and remain readable in gray.
    fig.legend(handles=[
        Line2D([], [], marker="o", color=BLUE, markerfacecolor="white", markersize=4.5, linestyle="none", label="Success"),
        Line2D([], [], marker="^", color=ORANGE, markersize=4.5, linestyle="none", label="Failure"),
    ], loc="upper right", bbox_to_anchor=(0.982, 0.890), ncol=2, frameon=False,
       handletextpad=0.35, columnspacing=1.0, borderaxespad=0.0, fontsize=8.0)
    figure_text(fig, 0.100, 0.458, "Area ∝ coincident trial count. Broken certificate axis; separate horizontal scales.", fontsize=8.0)

    roc = fig.add_axes([0.100, 0.125, 0.300, 0.255])
    hist = fig.add_axes([0.540, 0.125, 0.435, 0.255])
    figure_text(fig, 0.100, 0.423, "(b)  Cell-level failure ROC", fontweight="bold", fontsize=9.5)
    figure_text(fig, 0.540, 0.423, "(c)  Unbounded solver failures", fontweight="bold", fontsize=9.5)
    for ax in [roc, hist]:
        clean_axis(ax)
    roc.plot([0, 1], [0, 1], color="#999999", linestyle=(0, (3, 3)), linewidth=0.8, zorder=1)
    roc.plot(s["fpr"], s["tpr"], color=BLUE, linewidth=1.7, zorder=3)
    roc.plot(s["op_fpr"], s["op_tpr"], marker="D", markersize=5.1, markeredgecolor=INK,
             markerfacecolor="white", linewidth=0, zorder=5)
    roc.set(xlim=(-0.02, 1.025), ylim=(-0.02, 1.035), xlabel="False-positive rate", ylabel="True-positive rate")
    roc.set_xticks(np.linspace(0, 1, 3))
    roc.set_yticks(np.linspace(0, 1, 3))
    roc.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    roc.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    roc.text(0.38, 0.53, f"in-sample AUC\n{s['auc']:.2f}", transform=roc.transAxes,
             fontsize=9.2, fontweight="bold", linespacing=1.3,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    roc.text(0.38, 0.22, f"Positive class: failure\ncell success ≤ {s['positive_rate']:.0%}", transform=roc.transAxes,
             fontsize=8.0, linespacing=1.5)
    roc.annotate(rf"$\tau = {s['tau']:g}$", xy=(s["op_fpr"], s["op_tpr"]), xytext=(0.14, 0.82),
                 textcoords="axes fraction", fontsize=8.3,
                 arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.7))

    # Linear, equal-width count bins displayed against logarithmic error.
    # The width is a display choice, numerically equal to the supplied error cutoff.
    bin_width = s["success_limit"]
    edges = np.arange(math.floor(s["hist"].min() / bin_width) * bin_width,
                      (math.ceil(s["hist"].max() / bin_width) + 1) * bin_width, bin_width)
    frequencies, _, _ = hist.hist(s["hist"], bins=edges, color="#A8A8A8", edgecolor=INK,
                                   linewidth=0.45, zorder=3)
    assert int(frequencies.sum()) == len(s["hist"])
    hist.set_xscale("log")
    hist.set_xlim(s["hist"].min() * 0.80, edges[-1] * 1.08)
    hist.set_ylim(0, max(frequencies) * 1.70)
    hist.set_xlabel(r"Whole-model pose error $e_{\mathrm{tte}}$ (mm)")
    hist.set_ylabel("Failed trials", labelpad=5)
    hist.set_xticks([1, 2, 5, 10, 20, 40])
    hist.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    hist.xaxis.set_minor_locator(NullLocator())
    hist.set_yticks([0, 10, 20, 30])
    hist.text(0.02, 0.97, f"{len(s['hist'])} failures; {bin_width:g} mm bins", transform=hist.transAxes, va="top", fontsize=8.0)
    pitches = s["ctx"]["cad_pitches_mm"]
    for (name, pitch), xpos in zip(pitches.items(), [0.20, 0.78]):
        hist.axvline(pitch, ymin=0, ymax=0.60, color=ORANGE,
                    linewidth=0.9, linestyle=(0, (3, 2)), zorder=4)
        hist.annotate(f"{pitch:g} mm\n{name.replace('_', ' ')}",
                      xy=(pitch, max(frequencies) * 1.02),
                      xytext=(xpos, 0.82), textcoords="axes fraction",
                      ha="center", va="top", fontsize=8.0, linespacing=1.35,
                      arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8,
                                      connectionstyle="arc3", relpos=(0.5, 0), shrinkA=3, shrinkB=0))
    repeat_value, repeat_count = Counter(s["hist"]).most_common(1)[0]
    repeat_bin = np.searchsorted(edges, repeat_value, side="right") - 1
    hist.annotate(f"{repeat_count} exact repeats\nat {repeat_value:.2f} mm",
                  xy=(repeat_value, frequencies[repeat_bin]), xytext=(0.04, 0.48),
                  textcoords="axes fraction", ha="left", va="top", fontsize=8.0,
                  linespacing=1.35,
                  arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.7,
                                  connectionstyle="arc3", relpos=(1, 0), shrinkA=3))
    figure_text(fig, 0.100, 0.052,
                f"τ = {s['tau']:g}: supplied LOPO threshold; training-fold thresholds degenerate in {s['absent_folds']} of {s['total_folds']} folds.", fontsize=8.0)
    figure_text(fig, 0.100, 0.027,
                "Dashed orange lines: CAD inter-feature pitches; reference locations, not exact error modes.", fontsize=8.0)
    return fig


def audit_figure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    tight = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    texts = []
    outside = []
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        # Annotation.get_window_extent includes arrows; audit text separately.
        box = Text.get_window_extent(t, renderer=renderer)
        if not np.all(np.isfinite(box.extents)) or box.width == 0 or box.height == 0:
            continue
        texts.append((t.get_text(), box))
        if box.x0 < 0 or box.y0 < 0 or box.x1 > canvas.x1 or box.y1 > canvas.y1:
            outside.append(t.get_text())
    overlaps = []
    for (a, ba), (b, bb) in itertools.combinations(texts, 2):
        intersection = Bbox.intersection(ba, bb)
        if intersection is not None and intersection.width > 0.6 and intersection.height > 0.6:
            overlaps.append((a, b, round(intersection.width * intersection.height, 1)))
    tick_sizes = [t.get_fontsize() for ax in fig.axes for t in ax.get_xticklabels() + ax.get_yticklabels()
                  if t.get_visible() and t.get_text()]
    bbox_ok = (tight.x0 >= 0 and tight.y0 >= 0 and tight.x1 <= canvas.x1 and tight.y1 <= canvas.y1)
    result = dict(canvas_px=tuple(map(int, canvas.size)), tight_bbox_px=tuple(round(v, 1) for v in tight.extents),
                  bbox_inside_canvas=bool(bbox_ok), text_outside=outside, text_overlap_count=len(overlaps),
                  text_overlaps=overlaps, min_tick_label_pt=min(tick_sizes), text_count=len(texts))
    print("LAYOUT AUDIT:", json.dumps(result, ensure_ascii=False))
    return result


def make_caption(s):
    p = s["ctx"]["cad_pitches_mm"]
    return (
        f"Fig. 12. Single-view occlusion: (a) the offline, CAD-only smallest eigenvalue of the visible-subset Fisher "
        f"information matrix versus symmetry-aware whole-model RMS pose error for {s['n_trials']} "
        f"{s['ctx']['arm']} trials across {s['n_cells']} cells (success < {s['success_limit']:g} mm; "
        f"marker area encodes coincident trial count; the broken log certificate axis uses separate horizontal scales). "
        f"(b) Cell-level ROC (positive class: failure, defined as cell success rate at or below {s['positive_rate']:.0%}; "
        f"in-sample AUC = {s['auc']:.2f}), with the supplied LOPO threshold τ = {s['tau']:g} marked; "
        f"training-fold threshold selection degenerates to machine zero in {s['absent_folds']} of {s['total_folds']} folds. "
        f"(c) All {len(s['hist'])} unbounded-solver failure errors are shown with the {p['shoulder_offset']:g} mm shoulder offset "
        f"and {p['end_plane_gap']:g} mm end-plane gap marked as CAD inter-feature pitches, whose correspondence "
        f"with the measured clusters is approximate. "
        f"The certificate provides an a-priori flag supported by predictive association on this grid, while repeated errors "
        f"are compatible with discrete attraction basins; these observations establish neither out-of-sample validity nor causal proof."
    )


def main():
    s = read_data()
    fig = build_figure(s)
    audit = audit_figure(fig)
    fig.savefig(OUTPUT, dpi=DPI, metadata={"Title": "Fig. 12: single-view occlusion", "Description": make_caption(s)})
    raw = OUTPUT.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    dimensions = struct.unpack(">II", raw[16:24])
    assert dimensions == (round(WIDTH * DPI), round(HEIGHT * DPI))
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == s["source_hash"]
    print(f"DATA AUDIT: in-sample AUC = {s['auc']:.6f}; positive class = failure; "
          f"{s['positive'].sum()}/{s['n_cells']} positive cells; "
          f"threshold TPR = {s['op_tpr']:.6f}, FPR = {s['op_fpr']:.6f}")
    print("CAD reference exact matches:", {k: int(np.sum(s["hist"] == v)) for k, v in s["ctx"]["cad_pitches_mm"].items()})
    print(f"Final pixel dimensions: {dimensions[0]} × {dimensions[1]} ({WIDTH:g} × {HEIGHT:g} in at {DPI} dpi)")
    print(f"PNG file size: {len(raw):,} bytes ({len(raw) / 1024:.1f} KiB)")
    print("Suggested manuscript caption:\n" + make_caption(s))
    plt.close(fig)
    assert audit["bbox_inside_canvas"] and not audit["text_outside"], "Clipped figure elements"
    assert audit["min_tick_label_pt"] >= MIN_TICK_SIZE, "Tick labels too small"
    assert not audit["text_overlaps"], "Text overlap requires a layout revision"


if __name__ == "__main__":
    main()
