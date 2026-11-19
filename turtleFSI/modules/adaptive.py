"""Opt-in convergence-based stepping for the serial reference-flag problem.

No temporal error estimate is implied. Rejected trials never reach output or
post_solve hooks. The fixed reference mesh itself is not moved by this problem.
"""
import csv
import math
from pathlib import Path


class AdaptiveController:
    def __init__(self, dt, minimum, maximum, output_start, output_interval, end):
        values = (dt, minimum, maximum, output_start, output_interval, end)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("Adaptive times and bounds must be finite and positive")
        if not minimum <= dt <= maximum:
            raise ValueError("Require adaptive_dt_min <= initial dt <= adaptive_dt_max")
        self.minimum, self.maximum = minimum, maximum
        self.proposal = dt
        self.start, self.interval, self.end = output_start, output_interval, end
        self.output_index = 0
        self.easy_steps = 0

    @property
    def next_output(self):
        return self.start + self.output_index * self.interval

    def step(self, time):
        return min(self.proposal, self.end-time, self.next_output-time)

    def reject(self, attempted):
        self.easy_steps = 0
        if attempted <= self.minimum * (1 + 1e-10):
            raise RuntimeError("Adaptive step failed at minimum dt (or a smaller output-boundary step)")
        self.proposal = max(self.minimum, attempted / 2)

    def accept(self, time, iterations):
        output = math.isclose(time, self.next_output, rel_tol=0, abs_tol=1e-10)
        if output:
            self.output_index += 1
        self.easy_steps = self.easy_steps + 1 if iterations <= 5 else 0
        if self.easy_steps >= 3:
            self.proposal = min(self.maximum, 1.25*self.proposal)
            self.easy_steps = 0
        return output


def configure(ns):
    """Reject unsupported histories/hooks before creating run outputs."""
    from dolfin import MPI
    if (ns['problem'] != 'TF_fsi_reference' or MPI.size(MPI.comm_world) != 1
            or ns['theta'] != 1 or ns['d_deg'] != 2
            or ns['extrapolation'] != 'laplace' or ns['extrapolation_sub_type'] != 'paper'
            or ns['solver'] != 'newtonsolver' or ns['fluid'] != 'fluid' or ns['solid'] != 'solid'):
        raise ValueError("Adaptive mode supports serial TF_fsi_reference, theta=1, CG2 displacement, fluid/solid/newtonsolver and laplace/paper only")
    if ns['restart_folder'] is not None or ns['t'] != 0:
        raise ValueError("Adaptive mode currently requires a fresh run from time zero")
    maximum = ns['dt'] if ns['adaptive_dt_max'] is None else ns['adaptive_dt_max']
    return AdaptiveController(ns['dt'], ns['adaptive_dt_min'], maximum,
                               ns['adaptive_output_start'], ns['adaptive_output_interval'], ns['T'])


def advance(ns, controller):
    """Retry one step, restoring all solution history and inlet time on rejection."""
    import numpy as np
    old_time, old_dt = ns['t'], ns['dt']
    saved = {key: value.vector().get_local().copy() for key, value in ns['dvp_'].items()}
    log = Path(ns['results_folder']) / 'adaptive_steps.csv'

    def restore():
        for key, values in saved.items():
            vector = ns['dvp_'][key].vector()
            vector.set_local(values)
            vector.apply('insert')
        ns['t'], ns['dt'] = old_time, old_dt
        ns['k'].assign(old_dt)
        ns['dvp_res'].vector().zero()
        ns.update(ns['pre_solve'](**ns) or {})
        ns['b'] = None

    while True:
        attempted = controller.step(old_time)
        if attempted <= 0:
            raise RuntimeError("Adaptive controller cannot advance time")
        ns['dt'], ns['t'] = attempted, old_time + attempted
        ns['k'].assign(attempted)
        iterations, minimum_j, minimum_jk = None, None, None
        reason = ''
        try:
            ns.update(ns['pre_solve'](**ns) or {})
            # A changed k invalidates the previous Jacobian, including after rejection.
            arguments = dict(ns, recompute_tstep=1)
            ns.update(ns['newtonsolver'](**arguments))
            iterations = ns['newton_iterations']
            if not ns['newton_converged']:
                raise RuntimeError("Newton iteration limit reached without residual convergence")
            if not np.isfinite(ns['dvp_']['n'].vector().get_local()).all():
                raise RuntimeError("Non-finite solution")
            minimum_j, minimum_jk = ns['jacobian_monitor'].minimum()
            if not (minimum_j > 0 and minimum_jk > 0):
                raise RuntimeError("Non-positive deformation Jacobian")
        except (RuntimeError, ValueError) as error:
            reason = str(error)
        with log.open('a', newline='') as handle:
            writer = csv.writer(handle)
            if handle.tell() == 0:
                writer.writerow(['time_start', 'time_trial', 'dt', 'accepted', 'newton_iterations', 'min_J', 'min_JK', 'reason'])
            writer.writerow([old_time, old_time+attempted, attempted, not reason,
                             iterations, minimum_j, minimum_jk, reason])
        if not reason:
            return controller.accept(ns['t'], iterations)
        restore()
        print('Adaptive rejection at t=%g, dt=%g: %s' % (old_time+attempted, attempted, reason), flush=True)
        try:
            controller.reject(attempted)
        except RuntimeError as error:
            raise RuntimeError('%s; last rejection: %s' % (error, reason)) from error
