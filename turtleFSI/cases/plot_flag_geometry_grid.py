"""Plot saved initial geometry checks; no FSI simulation or remeshing."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Patch
from matplotlib.backends.backend_pdf import PdfPages

from turtleFSI.cases.flag_geometry import signed_double_area, triangle_quality


def plot_grid(directory):
    directory = Path(directory)
    reports = json.loads((directory / "validation.json").read_text())
    output = directory / "geometry_plots"
    output.mkdir(exist_ok=True)
    radii = sorted({r["radius"] for r in reports.values()})
    lengths = sorted({r["flag_length"] for r in reports.values()})
    records = {}
    for name, report in reports.items():
        with np.load(directory / (name + ".npz"), allow_pickle=False) as saved:
            x, cells = saved["mapped_coordinates"], saved["cells"]
            ref, tags = saved["reference_coordinates"], saved["cell_markers"]
        ratio = signed_double_area(x, cells)/signed_double_area(ref, cells)
        quality = triangle_quality(x, cells)
        np.testing.assert_allclose(ratio.min(), report["min_cell_area_ratio"])
        np.testing.assert_allclose(quality.min(), report["min_triangle_quality"])
        assert np.all(ratio > 0) and set(tags) == {1, 2}
        records[report["radius"], report["flag_length"]] = (x, cells, tags, ratio, quality)
    if len(records) != len(radii)*len(lengths):
        raise ValueError("Expected a complete radius-length grid")

    colors = {1: "#d9edf7", 2: "#ee983d"}
    legend = [Patch(facecolor=colors[1], label="Fluid domain"),
              Patch(facecolor=colors[2], label="Deformable solid flag"),
              Patch(facecolor="#bcbcbc", label="Rigid circle (excluded from FSI mesh)")]

    def draw(ax, radius, length, full=False, mark=False):
        x, cells, tags, ratio, quality = records[radius, length]
        ax.add_patch(Circle((.2, .2), radius, color="#bcbcbc", zorder=0))
        ax.add_collection(PolyCollection(x[cells], facecolors=[colors[int(t)] for t in tags],
                                        edgecolors="#416575", linewidths=.25, zorder=1))
        if mark:
            for index, color in ((np.argmin(ratio), "#d62728"),
                                 (np.argmin(quality), "#8b008b")):
                p = x[cells[index]]
                ax.plot(*np.vstack((p, p[0])).T, color=color, lw=1.8, zorder=3)
        ax.set(xlim=(0, 2.5 if full else .95), ylim=(0, .41), aspect="equal")
        ax.set_title(f"R={radius:.5g} m, length={length:.4g} m\n"
                     f"min K={ratio.min():.3f}; min quality={quality.min():.3f}", fontsize=10)
        ax.tick_params(labelsize=8)

    fig, axes = plt.subplots(len(radii), len(lengths), figsize=(22, 14), squeeze=False,
                             constrained_layout=True, sharex=True, sharey=True)
    for i, radius in enumerate(radii):
        for j, length in enumerate(lengths):
            draw(axes[i, j], radius, length)
            if j == 0:
                axes[i, j].set_ylabel("y (m)")
            if i == len(radii)-1:
                axes[i, j].set_xlabel("x (m)")
    fig.suptitle("Initial parameterized geometries T(X): updated map v2\n"
                 "Radius increases downwards; flag length increases to the right. "
                 "Same spatial scale in every panel; outlet at x=2.5 m is outside this zoom.", fontsize=16)
    fig.legend(handles=legend, loc="outside lower center", ncol=3)
    fig.savefig(output / "all_25_meshes.png", dpi=180)
    fig.savefig(output / "all_25_meshes.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (radius, length) in zip(axes.flat, [(radii[0], lengths[0]),
            (radii[0], lengths[-1]), (radii[-1], lengths[0]), (radii[-1], lengths[-1])]):
        draw(ax, radius, length, mark=True)
        ax.set(xlabel="x (m)", ylabel="y (m)")
    fig.suptitle("Four parameter-range corners: baseline, length-only, radius-only, combined\n"
                 "Red outline: smallest area ratio K. Purple outline: lowest triangle quality.")
    fig.legend(handles=legend, loc="outside lower center", ncol=3)
    fig.savefig(output / "four_corners.png", dpi=180)
    plt.close(fig)

    with PdfPages(output / "individual_meshes.pdf") as pdf:
        for radius in radii:
            for length in lengths:
                fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
                draw(axes[0], radius, length, full=True)
                draw(axes[1], radius, length, mark=True)
                for ax in axes:
                    ax.set(xlabel="x (m)", ylabel="y (m)")
                fig.suptitle("Full channel above; circle/flag detail below. Initial mesh, before physical motion.\n"
                             "Red: minimum K cell; purple: minimum quality cell.")
                fig.legend(handles=legend, loc="outside lower center", ncol=3)
                pdf.savefig(fig)
                plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, metric, title in zip(axes, (3, 4), ("Minimum area ratio K (guard: 0.20)",
                                                "Minimum triangle quality (guard: 0.02)")):
        values = np.array([[records[r, l][metric].min() for l in lengths] for r in radii])
        im = ax.imshow(values, cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(lengths)), xticklabels=[f"{l:.4g}" for l in lengths],
               yticks=range(len(radii)), yticklabels=[f"{r:.5g}" for r in radii],
               xlabel="Flag length (m)", ylabel="Circle radius (m)", title=title)
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if im.norm(value)<.55 else "black")
        fig.colorbar(im, ax=ax)
    fig.suptitle("Initial mesh diagnostics: smaller values mean less margin to the guards")
    fig.savefig(output / "quality_grid.png", dpi=180)
    plt.close(fig)
    print(f"Plotted {len(records)} saved meshes in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_grid(parser.parse_args().directory)
