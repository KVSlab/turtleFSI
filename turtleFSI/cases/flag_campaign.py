"""Prepare and validate the v2 flag training campaign; no plotting required."""
import argparse
import json
from pathlib import Path

import numpy as np

from turtleFSI.cases.flag_geometry import DEFAULT_MAPPING


def case_plan(dt=.01, stride=5, plan="training16"):
    if plan == "grid25":
        return [dict(name=f"radius_{i:02d}_length_{j:02d}", radius=radius,
                     length=length, role="training", dt=dt, stride=stride)
                for i, radius in enumerate((.05, .05625, .0625, .06875, .075))
                for j, length in enumerate((.35, .3875, .425, .4625, .5))]
    if plan != "training16":
        raise ValueError("Unknown campaign plan: " + plan)
    cases = []
    for i, radius in enumerate((.05, .0625, .075)):
        for j, length in enumerate((.35, .40, .45, .50)):
            cases.append(dict(name=f"train_r{i}_l{j}", radius=radius, length=length,
                              role="training", dt=dt, stride=stride))
    for i, (radius, length) in enumerate(((.05625, .425), (.06875, .475))):
        cases.append(dict(name=f"heldout_{i}", radius=radius, length=length,
                          role="heldout", dt=dt, stride=stride))
    for name, radius, length in (("train_r0_l0", .05, .35), ("train_r2_l3", .075, .50)):
        cases.append(dict(name=name+"_fine", radius=radius, length=length,
                          role="time_step_check", dt=dt/2, stride=2*stride,
                          compare_to=name))
    return cases


def prepare(root, end_time=30., dt=.01, stride=5, plan="training16"):
    from turtleFSI.cases.validate_flag_geometry import validate
    if not np.isfinite([end_time, dt]).all() or dt <= 0 or end_time < dt or stride < 1:
        raise ValueError("Require finite end time >= dt > 0 and positive snapshot stride")
    if not np.isclose(end_time/dt, round(end_time/dt), rtol=0, atol=1e-8):
        raise ValueError("End time must be a multiple of dt")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "campaign.json").exists() or (root / "results").exists():
        raise ValueError("Use a fresh campaign directory")
    cases = case_plan(dt, stride, plan)
    validate(root / "meshes", {c["name"]: (c["radius"], c["length"]) for c in cases})
    manifest = dict(mapping_version=DEFAULT_MAPPING, end_time=end_time, plan=plan, cases=cases)
    (root / "campaign.json").write_text(json.dumps(manifest, indent=2))
    (root / "cases.tsv").write_text("".join(
        f'{c["name"]}\t{c["radius"]}\t{c["length"]}\t{c["dt"]}\t{c["stride"]}\n'
        for c in cases))
    return manifest


def check_history(history, end_time, dt):
    required = ("time", "tip_dx", "tip_dy", "drag", "lift", "min_J", "min_JK")
    if history.dtype.names is None or not set(required) <= set(history.dtype.names):
        raise ValueError("Missing history columns")
    expected = dt*np.arange(1, int(round(end_time/dt))+2)
    if len(history) != len(expected):
        raise ValueError("Incomplete timestep history")
    np.testing.assert_allclose(history["time"], expected, rtol=0, atol=1e-8)
    if not all(np.isfinite(history[k]).all() for k in required):
        raise ValueError("Non-finite history")
    if min(history["min_J"].min(), history["min_JK"].min()) <= 0:
        raise ValueError("Non-positive time-dependent Jacobian")


def prepare_retry(source, root, retry_dt=None):
    """Prepare failed/unstarted cases in a fresh directory; never edit source."""
    from turtleFSI.cases.validate_flag_geometry import validate
    import copy
    source, root = Path(source).resolve(), Path(root).resolve()
    manifest = json.loads((source / "campaign.json").read_text())
    if manifest["mapping_version"] != DEFAULT_MAPPING:
        raise ValueError("Retry requires the same geometry-map version as the source")
    if retry_dt is not None and (not np.isfinite(retry_dt) or retry_dt <= 0):
        raise ValueError("Retry timestep must be finite and positive")
    if source == root or (root / "campaign.json").exists() or (root / "results").exists():
        raise ValueError("Retry output must be a fresh directory")
    pending, skipped = [], []
    for original in manifest["cases"]:
        case = copy.deepcopy(original)
        exitfile = source / (case["name"] + ".exitcode")
        code = int(exitfile.read_text()) if exitfile.exists() else None
        if code == 0:
            history = np.atleast_1d(np.genfromtxt(
                source / "results" / case["name"] / "reference_history.csv", delimiter=",", names=True))
            check_history(history, manifest["end_time"], case["dt"])
            skipped.append(case["name"])
            continue
        if code is not None and retry_dt is not None:
            if retry_dt > case["dt"]:
                raise ValueError("Retry timestep must not exceed the failed case timestep")
            interval = case["dt"] * case["stride"]
            stride = int(round(interval / retry_dt))
            if stride < 1 or not np.isclose(stride*retry_dt, interval):
                raise ValueError("Retry timestep must divide the saved snapshot interval")
            if not np.isclose(manifest["end_time"]/retry_dt, round(manifest["end_time"]/retry_dt)):
                raise ValueError("End time must be a multiple of retry timestep")
            case.update(previous_dt=case["dt"], dt=retry_dt, stride=stride)
        case["retry_reason"] = "failed" if code is not None else "not_started_or_interrupted"
        pending.append(case)
    if not pending:
        raise ValueError("No failed or unstarted cases remain")
    root.mkdir(parents=True, exist_ok=True)
    validate(root / "meshes", {c["name"]: (c["radius"], c["length"]) for c in pending})
    result = dict(manifest, cases=pending, source_campaign=str(source),
                  skipped_completed=skipped, restart_policy="from_time_zero")
    (root / "campaign.json").write_text(json.dumps(result, indent=2))
    (root / "end_time.txt").write_text(str(result["end_time"]))
    (root / "cases.tsv").write_text("".join(
        f'{c["name"]}\t{c["radius"]}\t{c["length"]}\t{c["dt"]}\t{c["stride"]}\n'
        for c in pending))
    print(f"Skipped {len(skipped)} completed cases; queued {len(pending)} cases from time zero.")
    return result


def validate_campaign(root):
    from turtleFSI.cases.validate_flag_snapshots import validate
    root = Path(root)
    manifest = json.loads((root / "campaign.json").read_text())
    reports, histories = {}, {}
    for dt in sorted({c["dt"] for c in manifest["cases"]}):
        group = {c["name"]: (c["radius"], c["length"])
                 for c in manifest["cases"] if c["dt"] == dt}
        reports.update(validate(root, group))
    for case in manifest["cases"]:
        name = case["name"]
        folder = root / "results" / name
        if (root / f"{name}.exitcode").read_text().strip() != "0":
            raise ValueError(f"{name}: solver did not exit successfully")
        report = json.loads((folder / "Mesh/geometry_map.json").read_text())
        if report.get("mapping_version") != manifest["mapping_version"]:
            raise ValueError(f"{name}: wrong mapping version")
        history = np.atleast_1d(np.genfromtxt(folder / "reference_history.csv", delimiter=",", names=True))
        check_history(history, manifest["end_time"], case["dt"])
        with np.load(folder / "ROM/dvp_snapshots.npz", allow_pickle=False) as data:
            # Export starts at solver counter zero, i.e. t=dt.
            expected = history["time"][::case["stride"]]
            np.testing.assert_allclose(data["time"], expected, rtol=0, atol=1e-8)
        histories[name] = history
        reports[name].update(role=case["role"], min_J=float(history["min_J"].min()),
                             min_JK=float(history["min_JK"].min()))
    comparisons = {}
    for case in manifest["cases"]:
        if case["role"] != "time_step_check":
            continue
        if case["compare_to"] not in histories:
            continue  # A successful coarse case may reside in the source campaign.
        coarse, fine = histories[case["compare_to"]], histories[case["name"]]
        time = coarse["time"]
        mask = time <= manifest["end_time"] + 1e-9
        metrics = {}
        for field in ("tip_dx", "tip_dy", "drag", "lift"):
            a = coarse[field][mask]
            b = np.interp(time[mask], fine["time"], fine[field])
            rms = float(np.sqrt(np.mean((a-b)**2)))
            scale = float(np.sqrt(np.mean(b**2)))
            metrics[field] = dict(rms_difference=rms,
                                  relative_rms_difference=rms/scale if scale > 0 else None)
        comparisons[case["compare_to"]] = metrics
    summary = dict(status="representation_and_completion_passed",
                   accuracy_review_required=True, cases=reports, time_step_comparisons=comparisons)
    (root / "campaign_validation.json").write_text(json.dumps(summary, indent=2))
    # Explicit split files keep held-out and refinement data out of POD training.
    for role, filename in (("training", "training_runs.txt"), ("heldout", "heldout_runs.txt")):
        (root / filename).write_text("".join(
            str((root / "results" / c["name"]).resolve())+"\n"
            for c in manifest["cases"] if c["role"] == role))
    print("PASS: campaign complete; numerical accuracy still requires review.")
    if not comparisons:
        print("This campaign contains no time-step refinement comparisons or automatic held-out split.")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "retry"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--end-time", type=float, default=30.)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--plan", choices=("training16", "grid25"), default="training16")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--retry-dt", type=float)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory, args.end_time, args.dt, args.stride, args.plan)
    elif args.action == "retry":
        if args.resume_from is None:
            parser.error("retry requires --resume-from")
        prepare_retry(args.resume_from, args.directory, args.retry_dt)
    else:
        validate_campaign(args.directory)


if __name__ == "__main__":
    main()
