"""Inspect fixed-state ROM derivatives and algebra without advancing a rollout."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--basis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rollout', type=Path,
                        help='Saved reproduction.npz: inspect ROM predictions rather than FOM states')
    parser.add_argument('--times', type=float, nargs='+', default=[10.01, 11.01, 11.51])
    args = parser.parse_args()
    from turtleFSI.rom._pod_galerkin.fenics_monolithic import FEniCSMonolithicProblem
    from turtleFSI.rom._pod_galerkin.problem_overrides import replay_overrides
    from turtleFSI.rom._pod_galerkin.monolithic_pod import load_monolithic_pod_basis
    from turtleFSI.rom._pod_galerkin.offline_online import equilibrate_linear_system
    from dolfin import assemble, inner, grad, inv, tr, sqrt
    from turtleFSI.modules.common import F_, G_, J_, K_
    source = args.run_dir / 'ROM/dvp_snapshots.npz'
    overrides = replay_overrides(source)
    if overrides['problem'] != 'TF_fsi_reference' or float(overrides['theta']) != 1.:
        raise ValueError('This audit requires backward-Euler reference-flag data')
    problem = FEniCSMonolithicProblem.build('TF_fsi_reference', overrides=overrides)
    basis = load_monolithic_pod_basis(args.basis)
    phi = problem.constrain_increment_basis(basis.pod_basis.basis)
    mean = basis.pod_basis.mean[:, 0]
    free = problem.free_dofs()
    fields = {f.name: np.arange(f.start, f.stop) for f in basis.fields}
    velocity = np.concatenate([fields['velocity'], fields.get('supremizer', np.array([], dtype=int))])
    pressure = fields['pressure']
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(run_dir=str(args.run_dir.resolve()), basis=str(args.basis.resolve()),
                  dt=float(overrides['dt']), advanced_steps=0, states=[])
    report['rollout'] = str(args.rollout.resolve()) if args.rollout else None
    report['history_note'] = ('Jacobian checks freeze previous state to the selected current state; '
                             'these are not the original rollout Newton matrices. Kinematic defects '
                             'use saved-interval secants versus endpoint velocity, not exact 0.01s defects.')
    rollout = np.load(args.rollout) if args.rollout else None
    ns = problem.namespace
    d, v = ns['d_']['n'], ns['v_']['n']
    shape = ns.get('shape_displacement')
    divergence = tr(grad(v)*inv(G_(d, shape))*inv(F_(d, shape)))
    volume_weight = K_(d, shape)*J_(d, shape)
    def defects(current, previous, interval):
        problem.set_state(current)
        problem.set_previous_state(previous)
        physical_divergence = sqrt(assemble(sum(volume_weight*divergence**2*ns['dx_f'][i]
                                               for i in range(len(ns['dx_f_id_list'])))))
        mismatch = (d-ns['d_']['n-1'])/interval-v
        mismatch_norm = sqrt(assemble(sum(K_(d, shape)*inner(mismatch, mismatch)*ns['dx_s'][i]
                                         for i in range(len(ns['dx_s_id_list'])))))
        return dict(fluid_divergence_L2=float(physical_divergence),
                    solid_secant_kinematic_L2=float(mismatch_norm), saved_interval=float(interval))
    with np.load(source) as snapshots:
        for requested in args.times:
            data_times = rollout['time'] if rollout is not None else snapshots['time']
            index = int(np.argmin(abs(data_times-requested)))
            if abs(data_times[index]-requested) > 1e-8:
                raise ValueError('Requested time is not saved: %g' % requested)
            previous = rollout['prediction'][:, index] if rollout is not None else snapshots['dvp'][:, index]
            physical_defects = None
            if rollout is not None:
                if index == 0:
                    source_index = int(np.argmin(abs(snapshots['time']-requested)))
                    before_rom = before_fom = snapshots['dvp'][:, source_index-1]
                    interval = float(requested-snapshots['time'][source_index-1])
                else:
                    before_rom, before_fom = rollout['prediction'][:, index-1], rollout['reference'][:, index-1]
                    interval = float(data_times[index]-data_times[index-1])
                physical_defects = dict(rom=defects(previous, before_rom, interval),
                                        fom=defects(rollout['reference'][:, index], before_fom, interval))
            problem.set_previous_state(previous)
            problem.set_previous_previous_state(previous)
            problem.set_time(float(requested+overrides['dt']))
            a = np.linalg.lstsq(phi[free], previous[free]-mean[free], rcond=None)[0]

            def residual(coefficients):
                state = problem.apply_state_bcs(mean+phi@coefficients)
                return problem.projected_residual(phi, state).ravel()

            state = problem.apply_state_bcs(mean+phi@a)
            J = problem.projected_jacobian(phi, state)
            R = residual(a)
            scaled, _, _, _ = equilibrate_linear_system(J, -R)
            singular = np.linalg.svd(J, compute_uv=False)
            scaled_singular = np.linalg.svd(scaled, compute_uv=False)
            coupling = J[np.ix_(pressure, velocity)]
            coupling_singular = np.linalg.svd(coupling, compute_uv=False)
            checks = []
            rng = np.random.default_rng(0)
            for field, columns in fields.items():
                direction = np.zeros(len(a))
                direction[columns] = rng.normal(size=len(columns))
                direction /= np.linalg.norm(direction)
                expected = J@direction
                errors = []
                for epsilon in (1e-3, 1e-4, 1e-5, 1e-6):
                    numerical = (residual(a+epsilon*direction)-residual(a-epsilon*direction))/(2*epsilon)
                    errors.append(dict(epsilon=epsilon, relative_error=float(np.linalg.norm(numerical-expected)/
                                  max(np.linalg.norm(expected), np.linalg.norm(numerical), 1e-30))))
                checks.append(dict(field=field, finite_difference=errors))
            record = dict(initial_time=requested, evaluation_time=float(requested+overrides['dt']),
                          physical_defects=physical_defects,
                          projected_residual_norm=float(np.linalg.norm(R)),
                          jacobian_singular_values=singular.tolist(),
                          equilibrated_singular_values=scaled_singular.tolist(),
                          equilibrated_rank_at_rcond_1e12=int(np.sum(scaled_singular > 1e-12*scaled_singular[0])),
                          num_modes=len(a), continuity_coupling_singular_values=coupling_singular.tolist(),
                          derivative_checks=checks,
                          note='Coupling singular values use coefficient norms; not a physical inf-sup constant.')
            report['states'].append(record)
            np.savez_compressed(args.output_dir/('operators_%g.npz' % requested),
                                jacobian=J, residual=R, scaled_jacobian=scaled)
            (args.output_dir/'audit.json').write_text(json.dumps(report, indent=2))
            print('Audited t=%g; scaled rank %d/%d' %
                  (requested, record['equilibrated_rank_at_rcond_1e12'], len(a)), flush=True)
    print('Saved audit to %s; no timesteps advanced' % args.output_dir, flush=True)
    if rollout is not None:
        rollout.close()


if __name__ == '__main__':
    main()
