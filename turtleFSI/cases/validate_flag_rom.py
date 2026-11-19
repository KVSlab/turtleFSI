"""Baseline intrusive POD-Galerkin reproduction and held-out time prediction.

Uses the checkpoint timestep, substeps between saved snapshots, and does not
feed reference states back into a rollout. Outputs go to a separate directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evaluation_windows(times, args):
    """Select saved evaluation endpoints independently of the training cutoff."""
    def index_at(t):
        i = int(np.argmin(abs(times-t)))
        require(abs(times[i]-t) < 1e-8, "Requested time is not a saved snapshot: %g" % t)
        return i
    require(np.isfinite(args.train_end), "Training cutoff must be finite")
    if getattr(args, "reproduction_only", False):
        require(not args.prediction_only and args.prediction_start is None,
                "Reproduction-only cannot be combined with prediction options")
        first, end = index_at(args.reproduction_start), index_at(args.prediction_end)
        require(first < end and times[end] <= args.train_end + 1e-8,
                "Need reproduction start < end <= training cutoff")
        phases = (("reproduction", first, end),)
    elif args.prediction_only:
        prediction_start = args.train_end if args.prediction_start is None else args.prediction_start
        require(prediction_start >= args.train_end, "Prediction must start at or after training cutoff")
        first, end = index_at(prediction_start), index_at(args.prediction_end)
        require(first < end, "Prediction start must precede prediction end")
        phases = (("prediction", first, end),)
    else:
        require(args.prediction_start is None, "--prediction-start requires --prediction-only")
        train_end = index_at(args.train_end)
        start = index_at(args.reproduction_start)
        end = index_at(args.prediction_end)
        require(start < train_end < end, "Need reproduction start < training end < prediction end")
        phases = (("reproduction", start, train_end), ("prediction", train_end, end))
    return phases


def enrich_velocity_with_displacement(problem, basis, phi):
    """Transport continuous CG displacement modes into the velocity space.

    The existing velocity span is preserved. Only independent residual columns
    are added; all Dirichlet increments remain zero. Requires matching elements.
    """
    D, dm = problem.DVP.sub(0).collapse(collapsed_dofs=True)
    V, vm = problem.DVP.sub(1).collapse(collapsed_dofs=True)
    require(D.ufl_element() == V.ufl_element(), "Kinematic enrichment requires matching d/u elements")
    require(np.array_equal(D.tabulate_dof_coordinates(), V.tabulate_dof_coordinates()),
            "Kinematic enrichment requires matching collapsed d/u order")
    di = np.array([dm[i] for i in range(D.dim())])
    vi = np.array([vm[i] for i in range(V.dim())])
    field = next(f for f in basis.fields if f.name == "displacement")
    candidates = np.zeros((phi.shape[0], field.stop-field.start))
    candidates[vi] = phi[di, field.start:field.stop]
    candidates = problem.constrain_increment_basis(candidates)
    # Work in velocity rows so the displacement/pressure blocks cannot enter.
    columns = np.flatnonzero(np.linalg.norm(phi[vi], axis=0) > 0)
    existing = phi[np.ix_(vi, columns)]
    residual = candidates[vi]-existing@np.linalg.lstsq(existing, candidates[vi], rcond=1e-12)[0]
    U, s, _ = np.linalg.svd(residual, full_matrices=False)
    keep = s > 1e-10
    extra = np.zeros((phi.shape[0], int(keep.sum())))
    extra[vi] = U[:, keep]
    return np.column_stack((phi, extra)), extra.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-end", type=float, default=3.01)
    parser.add_argument("--reproduction-start", type=float, default=2.01)
    parser.add_argument("--prediction-end", type=float, default=3.51)
    parser.add_argument("--prediction-only", action="store_true",
                        help="Skip reproduction and evaluate only the prediction window")
    parser.add_argument("--prediction-start", type=float,
                        help="Saved start time for prediction-only evaluation, independent of training cutoff")
    parser.add_argument("--modes", type=int, default=24)
    parser.add_argument("--energy-threshold", type=float,
                        help="Select modes by retained energy, without the default 24-mode cap")
    parser.add_argument("--reproduction-only", action="store_true",
                        help="Evaluate reproduction-start through prediction-end within training")
    parser.add_argument("--basis", type=Path, help="Reuse a previously trained basis; no training in this run")
    parser.add_argument("--identity-map", action="store_true",
                        help="Diagnostic: rebase onto the saved physical initial mesh; reuse a single-case basis")
    parser.add_argument("--training-run-dir", type=Path, nargs="+",
                        help="Training geometries; defaults to the evaluation run")
    parser.add_argument("--kinematic-enrichment", action="store_true",
                        help="Enrich velocity with transported displacement modes to support d_t=u")
    for field in ("displacement", "velocity", "pressure", "supremizer"):
        parser.add_argument("--"+field+"-modes", type=int)
    args = parser.parse_args()
    if args.energy_threshold is not None:
        require(0 < args.energy_threshold <= 1, "Energy threshold must be in (0, 1]")
    from turtleFSI.rom.compute_supremizers import compute_bundle
    from turtleFSI.rom._pod_galerkin.problem_overrides import replay_overrides
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
    from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
    from turtleFSI.rom._pod_galerkin.reduced_solvers import reduced_newton_solve_projected
    from turtleFSI.cases.flag_monitor import FlagJacobianMonitor
    from dolfin import Point

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = args.run_dir / "ROM/dvp_snapshots.npz"
    with np.load(source) as data:
        states, times = data["dvp"], data["time"]
        dofs = {name: data[key] for name, key in (("displacement", "displacement_dofs"),
                ("velocity", "velocity_dofs"), ("pressure", "fluid_pressure_dofs"))}
    overrides = replay_overrides(source)
    dt = float(overrides["dt"])
    require(overrides["problem"] == "TF_fsi_reference", "Expected reference-flag snapshots")
    require(float(overrides["theta"]) == 1., "This experiment uses backward Euler")
    phases = evaluation_windows(times, args)
    if args.identity_map:
        require(args.basis is not None and not args.kinematic_enrichment,
                "Identity-map comparison requires --basis and no kinematic enrichment")
        require(not args.training_run_dir, "Identity-map comparison uses a single-case basis")
    # Enrich training samples only: held-out states never enter the basis.
    training_runs = args.training_run_dir or [args.run_dir]
    training_counts = {}
    if args.basis is None:
        enriched_paths = []
        for number, run in enumerate(training_runs):
            with np.load(run / "ROM/dvp_snapshots.npz") as data:
                training = {key: data[key] for key in data.files}
            selected = training["time"] <= args.train_end + 1e-8
            require(selected.any(), "Training window contains no snapshots: %s" % run)
            training_counts[str(run.resolve())] = int(selected.sum())
            for key in ("dvp", "time", "counter"):
                training[key] = training[key][..., selected]
            train_path = out / ("training_snapshots_%d.npz" % number)
            np.savez_compressed(train_path, **training)
            enriched_paths.append(compute_bundle(run, train_path, out / ("training_supremizers_%d.npz" % number)))
        command = [sys.executable, "-m", "turtleFSI.rom._pod_galerkin.build_monolithic_pod_basis",
                   "--snapshots", *map(str, enriched_paths), "--output-dir", str(out / "basis"),
                   "--blocked-dvp", "--pressure-treatment", "supremizer"]
        if args.energy_threshold is None:
            command.extend(["--max-modes", str(args.modes)])
        else:
            command.extend(["--energy-threshold", str(args.energy_threshold)])
        for field in ("displacement", "velocity", "pressure", "supremizer"):
            cap = getattr(args, field+"_modes")
            if cap is not None:
                command.extend(["--"+field+"-modes", str(cap)])
        subprocess.run(command, check=True)
    basis_path = args.basis or out / "basis/monolithic_pod_basis.npz"
    basis = load_monolithic_pod_basis(basis_path)
    problem = FEniCSMonolithicProblem.build("TF_fsi_reference", overrides=overrides)
    require(problem.vector_size == states.shape[0], "Incompatible state dimensions")
    for i, name in enumerate(("displacement", "velocity")):
        require(np.array_equal(problem.DVP.sub(i).dofmap().dofs(), dofs[name]), "DoF map mismatch")
    phi = problem.constrain_increment_basis(basis.pod_basis.basis)
    identity_audit = None
    if args.identity_map:
        from turtleFSI.cases.flag_identity_check import build_identity_problem, compare_operators
        reference = problem
        problem = build_identity_problem(args.run_dir, overrides, reference)
        identity_audit = compare_operators(reference, problem, phi, states, times, phases[0][1], dt)
        (out / "identity_operator_check.json").write_text(json.dumps(identity_audit, indent=2))
        print("Identity-map residual/Jacobian equivalence passed", flush=True)
    added_kinematic_modes = 0
    if args.kinematic_enrichment:
        phi, added_kinematic_modes = enrich_velocity_with_displacement(problem, basis, phi)
    np.savez_compressed(out / "effective_basis.npz", basis=phi, mean=basis.pod_basis.mean)
    mean = basis.pod_basis.mean[:, 0]
    free = problem.free_dofs()
    monitor = FlagJacobianMonitor(problem.namespace["mesh"], problem.dvp_["n"].sub(0),
                                  problem.namespace["shape_displacement"])
    summary = dict(run_dir=str(args.run_dir.resolve()), dt=dt, train_end=float(args.train_end),
                   identity_map=args.identity_map, identity_operator_check=identity_audit,
                   basis=str(basis_path.resolve()), training_runs=[str(r.resolve()) for r in training_runs],
                   reused_basis=args.basis is not None,
                   geometry_held_out=args.run_dir.resolve() not in [r.resolve() for r in training_runs],
                   training_snapshot_counts=training_counts,
                   kinematic_enrichment=args.kinematic_enrichment, added_kinematic_modes=added_kinematic_modes,
                   training_snapshots=int(np.count_nonzero(times <= args.train_end + 1e-8)),
                   prediction_only=args.prediction_only,
                   full_dofs=problem.vector_size, modes=phi.shape[1],
                   method="intrusive projected FEniCS; full-order assembly, no hyper-reduction",
                   error_metric="Euclidean coefficient relative norm per field; active fluid pressure only",
                   phases={})
    comparisons = {}
    tip = Point(.6, .2)
    if args.identity_map:
        tip = Point(float(overrides['c_x'])+float(overrides['R'])+float(overrides['f_L']), float(overrides['c_y']))
    for phase, first, last in phases:
        selected_reference = states[:, first+1:last+1]
        best_coefficients = np.linalg.lstsq(phi[free], selected_reference[free]-mean[free, None], rcond=None)[0]
        best = mean[:, None] + phi @ best_coefficients
        for col, reference_index in enumerate(range(first+1, last+1)):
            problem.set_time(float(times[reference_index]))
            best[:, col] = problem.apply_state_bcs(best[:, col])
        projection_errors = {name: float(np.linalg.norm(best[indices]-selected_reference[indices]) /
                              np.linalg.norm(selected_reference[indices])) for name, indices in dofs.items()}
        previous = states[:, first].copy()
        coefficients = np.linalg.lstsq(phi[free], previous[free]-mean[free], rcond=None)[0][:, None]
        predictions, references, matched_times, logs = [], [], [], []
        tic = time.monotonic()
        failed = False
        for target in range(first+1, last+1):
            steps = int(round((times[target]-times[target-1])/dt))
            require(steps > 0 and abs(steps*dt-(times[target]-times[target-1])) < 1e-8,
                    "Snapshot interval must be an integer multiple of dt")
            for substep in range(1, steps+1):
                t = float(times[target-1] + substep*dt)
                problem.set_previous_state(previous)
                problem.set_previous_previous_state(previous)
                problem.set_time(t)
                def reconstruct(a):
                    return problem.apply_state_bcs(mean+phi@np.asarray(a).ravel())[:, None]
                result = reduced_newton_solve_projected(
                    initial_coefficients=coefficients,
                    projected_residual_callback=lambda a: problem.projected_residual(phi, reconstruct(a)[:, 0]),
                    projected_jacobian_callback=lambda a: problem.projected_jacobian(phi, reconstruct(a)[:, 0]),
                    reconstruct_callback=reconstruct, atol=1e-8, rtol=1e-9, max_iterations=20,
                    linear_solver="equilibrated-lstsq", linear_rcond=1e-12,
                    reuse_jacobian_after_line_search_failure=False)
                coefficients = result.coefficients
                previous = result.full_state[:, 0]
                problem.set_state(previous)
                min_J, min_JK = monitor.minimum()
                logs.append(dict(time=t, converged=bool(result.converged), iterations=int(result.iterations),
                                 jacobian_condition_numbers=list(result.jacobian_condition_numbers),
                                 jacobian_evaluations=int(result.jacobian_evaluations),
                                 residual=float(result.residual_norms[-1]), min_J=min_J, min_JK=min_JK,
                                 termination_reason=result.termination_reason))
                if not result.converged or min_J <= 0 or not np.isfinite(previous).all():
                    failed = True
                    break
            if failed:
                break
            predictions.append(previous.copy())
            references.append(states[:, target])
            matched_times.append(times[target])
            print("%s t=%.2f, Newton=%d, min J=%.6g" % (phase, t, result.iterations, min_J), flush=True)
        pred = np.array(predictions).T if predictions else np.empty((len(mean), 0))
        ref = np.array(references).T if references else np.empty((len(mean), 0))
        metrics = {}
        matched_projection_errors = {}
        for name, indices in dofs.items():
            denominator = np.linalg.norm(ref[indices])
            metrics[name] = float(np.linalg.norm(pred[indices]-ref[indices])/denominator) if denominator else None
            matched_projection_errors[name] = float(np.linalg.norm(
                best[indices, :len(matched_times)]-ref[indices])/denominator) if denominator else None
        tip_pred, tip_ref = [], []
        for column in range(pred.shape[1]):
            problem.set_state(pred[:, column])
            tip_pred.append(problem.dvp_["n"].sub(0)(tip))
            problem.set_state(ref[:, column])
            tip_ref.append(problem.dvp_["n"].sub(0)(tip))
        np.savez_compressed(out / (phase + ".npz"), time=matched_times, prediction=pred, reference=ref,
                            tip_prediction=tip_pred, tip_reference=tip_ref,
                            best_projection=best[:, :len(matched_times)])
        summary["phases"][phase] = dict(initial_time=float(times[first]), requested_end=float(times[last]),
             initialized_from="full-order state at phase start only", completed=not failed,
             matched_snapshots=len(matched_times), elapsed_seconds=time.monotonic()-tic,
             relative_errors=metrics, best_projection_errors=matched_projection_errors,
             requested_window_projection_errors=projection_errors, steps=logs)
        comparisons[phase] = (np.array(matched_times), np.array(tip_pred), np.array(tip_ref))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        if failed:
            print("Stopped after failed reduced solve; partial results saved", flush=True)
            break
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for phase, (t, pred, ref) in comparisons.items():
        if len(t):
            for i, ax in enumerate(axes):
                ax.plot(t, ref[:, i], label=phase+" FOM")
                ax.plot(t, pred[:, i], "--", label=phase+" ROM")
                ax.set_ylabel("Tip %s displacement (m)" % "xy"[i])
                ax.legend()
                ax.grid(alpha=.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out / "tip_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: {k: v for k, v in value.items() if k != "steps"}
                      for key, value in summary["phases"].items()}, indent=2))
    if any(not phase["completed"] for phase in summary["phases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
