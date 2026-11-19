"""Plot the reference flag histories after a parallel validation run."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    fields = ["tip_dx", "tip_dy", "drag", "lift", "min_J", "min_JK"]
    labels = ["Tip x displacement (m)", "Tip y displacement (m)",
              "Drag (2D force integral)", "Lift (2D force integral)", "Minimum J", "Minimum JK"]
    summary = {}
    for name in ("baseline", "longer_flag", "larger_circle"):
        path = args.run_directory / "results" / name / "reference_history.csv"
        history = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
        if len(history) == 0:
            raise ValueError("Empty history: %s" % path)
        finite = all(np.isfinite(history[field]).all() for field in ["time"] + fields)
        summary[name] = dict(steps=len(history), final_time=float(history["time"][-1]),
                             finite=bool(finite), min_J=float(np.nanmin(history["min_J"])),
                             min_JK=float(np.nanmin(history["min_JK"])))
        for ax, field, label in zip(axes.flat, fields, labels):
            ax.plot(history["time"], history[field], label=name)
            ax.set_ylabel(label)
            ax.grid(alpha=.3)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
        ax.axhline(0, color="black", linewidth=.7)
    axes[0, 0].legend()
    fig.tight_layout()
    output = args.run_directory / "validation_plots"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "histories.png", dpi=160)
    plt.close(fig)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Plot:", output / "histories.png")


if __name__ == "__main__":
    main()
