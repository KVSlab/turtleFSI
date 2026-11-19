"""Physical FOM/ROM comparison and actual-scale animation for saved 2-D FSI runs."""
from pathlib import Path
import argparse
import json
import numpy as np


def load_inputs(run_dir, rollout=None):
    run_dir = Path(run_dir).resolve()
    if rollout is None:
        candidates = [run_dir / 'analysis' / name / 'fenics_pod_galerkin_rollout.npz'
                      for name in ('validation_rollout', 'rollout')]
        candidates = [p for p in candidates if p.is_file()]
        if len(candidates) != 1:
            raise ValueError('Supply --rollout: expected exactly one analysis/validation_rollout or analysis/rollout archive')
        rollout = candidates[0]
    rollout = Path(rollout).resolve()
    settings = json.loads((run_dir / 'Checkpoint/default_variables.json').read_text())
    sources = ([rollout / name for name in ('reproduction.npz', 'prediction.npz')
                if (rollout / name).is_file()] if rollout.is_dir() else [rollout])
    if not sources:
        raise ValueError('No reproduction.npz or prediction.npz in rollout directory')
    blocks = []
    for source in sources:
        with np.load(source) as data:
            blocks.append((data['reference'], data['full'] if 'full' in data else data['prediction'], data['time']))
    reference = np.concatenate([b[0] for b in blocks], axis=1)
    reduced = np.concatenate([b[1] for b in blocks], axis=1)
    times = np.concatenate([b[2] for b in blocks])
    if (reference.ndim != 2 or reference.shape != reduced.shape or times.ndim != 1
            or len(times) < 2 or reference.shape[1] != len(times)
            or not all(np.isfinite(a).all() for a in (reference, reduced, times))
            or not np.all(np.diff(times) > 0)):
        raise ValueError('Expected matching finite DOF-by-time states and at least two strictly increasing times')
    summary_path = (rollout / 'summary.json' if rollout.is_dir() else
                    rollout.with_name(rollout.stem + '_summary.json'))
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return run_dir, rollout, settings, reference, reduced, times, summary


def limits(values, pad=.1):
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = max((hi-lo)*pad, 1e-9)
    return lo-margin, hi+margin


def relative_errors(squared_error, squared_norm):
    """Zero reference norms have undefined relative errors, represented by NaN."""
    result = np.full_like(squared_error, np.nan, dtype=float)
    np.divide(np.maximum(squared_error, 0), squared_norm, out=result, where=squared_norm > 0)
    return np.sqrt(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--rollout', type=Path, help='Saved NPZ or validate_flag_rom output directory')
    parser.add_argument('--output-dir', type=Path, help='Defaults to a physical_comparison folder beside the rollout directory')
    parser.add_argument('--tip', type=float, nargs=2, metavar=('X', 'Y'), help='Reference probe point; inferred for flag geometry')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--static-only', action='store_true')
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error('--fps must be positive')
    run, source, settings, fom, rom, times, summary = load_inputs(args.run_dir, args.rollout)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from dolfin import (Mesh, MeshFunction, HDF5File, VectorElement, FiniteElement,
                        MixedElement, FunctionSpace, Function, Measure, split, inner,
                        assemble, Point, MPI, VectorFunctionSpace, Identity, grad, det)
    if MPI.size(MPI.comm_world) != 1:
        raise ValueError('Run this postprocessor on one MPI rank')
    if not args.static_only and not FFMpegWriter.isAvailable():
        raise RuntimeError('ffmpeg is required for MP4 output; activate the FEniCS environment or use --static-only')
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(run / 'Mesh/mesh.h5'), 'r') as handle:
        handle.read(mesh, '/mesh', False)
        if mesh.geometry().dim() != 2 or mesh.topology().dim() != 2:
            raise ValueError('Physical four-panel comparison supports 2-D triangular FSI meshes')
        domains = MeshFunction('size_t', mesh, mesh.topology().dim())
        handle.read(domains, '/domains')
    cell = mesh.ufl_cell()
    elements = [VectorElement('CG', cell, settings['d_deg']),
                VectorElement('CG', cell, settings['v_deg']),
                FiniteElement('CG', cell, settings['p_deg'])]
    if settings.get('extrapolation') == 'biharmonic':
        elements.append(VectorElement('CG', cell, settings['d_deg']))
    space = FunctionSpace(mesh, MixedElement(elements))
    if space.dim() != fom.shape[0]:
        raise ValueError(f'Saved state dimension {fom.shape[0]} does not match source mesh/space {space.dim()}')
    shape = np.zeros_like(mesh.coordinates())
    weight = 1.0
    reference_flag = settings.get('problem') == 'TF_fsi_reference'
    if reference_flag:
        g = Function(VectorFunctionSpace(mesh, 'CG', 1))
        with HDF5File(mesh.mpi_comm(), str(run / 'Mesh/shape_displacement.h5'), 'r') as handle:
            handle.read(g, '/shape_displacement')
        shape = g.compute_vertex_values(mesh).reshape(2, -1).T
        weight = det(Identity(2) + grad(g))
    fluid_ids = [p['dx_f_id'] for p in settings['fluid_properties']]
    solid_ids = [p['dx_s_id'] for p in settings['solid_properties']]
    fluid, solid = np.isin(domains.array(), fluid_ids), np.isin(domains.array(), solid_ids)
    if not fluid.any() or not solid.any():
        raise ValueError('Source mesh must contain the configured fluid and solid domains')
    tip = args.tip
    if tip is None:
        if not all(key in settings for key in ('c_x', 'c_y', 'R', 'f_L')):
            parser.error('Supply --tip X Y for cases without flag geometry settings')
        tip = ([.6, .2] if reference_flag else
               [settings['c_x']+settings['R']+settings['f_L'], settings['c_y']])
    probe = Point(*tip)
    f, r = Function(space), Function(space)
    dx = Measure('dx', domain=mesh, subdomain_data=domains)
    solid_dx, fluid_dx = dx(solid_ids[0]), dx(fluid_ids[0])
    for domain_id in solid_ids[1:]: solid_dx = solid_dx + dx(domain_id)
    for domain_id in fluid_ids[1:]: fluid_dx = fluid_dx + dx(domain_id)
    ff, rr = split(f)[:3], split(r)[:3]
    measures = [solid_dx, fluid_dx, fluid_dx]
    norms = [inner(a, a)*weight*m for a, m in zip(ff, measures)]
    differences = [inner(b-a, b-a)*weight*m for a, b, m in zip(ff, rr, measures)]
    errors, displacements, speeds, tips = [], [[], []], [[], []], [[], []]
    for i in range(len(times)):
        for k, (state, data) in enumerate(((f, fom), (r, rom))):
            state.vector().set_local(data[:, i]); state.vector().apply('insert')
            d = state.sub(0, deepcopy=True)
            tips[k].append(d(probe))
            displacements[k].append(d.compute_vertex_values(mesh).reshape(2, -1).T)
            u = state.sub(1, deepcopy=True).compute_vertex_values(mesh).reshape(2, -1).T
            speeds[k].append(np.linalg.norm(u, axis=1))
        errors.append(relative_errors(np.array([assemble(e) for e in differences]),
                                      np.array([assemble(n) for n in norms])))
    errors, displacements, speeds, tips = map(np.asarray, (errors, displacements, speeds, tips))
    out = (args.output_dir or (source if source.is_dir() else source.parent.parent) / 'physical_comparison').resolve()
    out.mkdir(parents=True, exist_ok=True)
    tip_error = np.linalg.norm(tips[1]-tips[0], axis=1)
    np.savez_compressed(out/'physical_comparison.npz', time=times, errors=errors,
                        tip_fom=tips[0], tip_rom=tips[1])
    np.savetxt(out/'physical_errors.csv', np.column_stack([times, errors, tip_error]), delimiter=',',
               header='time,solid_displacement_relative_L2,fluid_velocity_relative_L2,fluid_pressure_relative_L2,tip_error_m', comments='')
    labels = ['Solid displacement', 'Fluid velocity', 'Fluid pressure']
    title = f'{run.name} • {times[0]:g}–{times[-1]:g} s'
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for j, axis in enumerate(axes[0]):
        for k, label in enumerate(('FOM', 'ROM')):
            axis.plot(times, tips[k,:,j], '-' if k == 0 else '--', label=label)
        axis.set(title=('Horizontal' if j == 0 else 'Vertical')+' tip displacement', xlabel='Time (s)', ylabel='Displacement (m)')
        axis.legend(); axis.grid(alpha=.2)
    for j, label in enumerate(labels):
        axes[1,0].plot(times, errors[:,j], label=label)
    axes[1,0].set(title='Relative spatial L2 error', xlabel='Time (s)', ylabel='||ROM − FOM|| / ||FOM||')
    axes[1,0].legend(); axes[1,0].grid(alpha=.2)
    axes[1,1].plot(times, tip_error)
    axes[1,1].set(title='Tip-position error', xlabel='Time (s)', ylabel='Error (m)')
    fig.suptitle(title)
    fig.savefig(out/'time_comparison.png', dpi=150); fig.savefig(out/'time_comparison.pdf'); plt.close(fig)
    coords, cells = mesh.coordinates(), mesh.cells()
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for j, (label, mask) in enumerate(zip(labels, (solid, fluid, fluid))):
        values = [s.sub(j, deepcopy=True).compute_vertex_values(mesh).reshape(1 if j == 2 else 2, -1).T for s in (f, r)]
        magnitudes = [v[:,0] if j == 2 else np.linalg.norm(v, axis=1) for v in values]
        error = np.linalg.norm(values[1]-values[0], axis=1)
        active = np.unique(cells[mask]); vmin = min(v[active].min() for v in magnitudes); vmax = max(v[active].max() for v in magnitudes)
        tri = mtri.Triangulation(coords[:,0], coords[:,1], cells, mask=~mask)
        for k, v in enumerate((*magnitudes, error)):
            ax = axes[j,k]
            artist = ax.tripcolor(tri, v, shading='gouraud', cmap='viridis' if k < 2 else 'magma',
                                  vmin=vmin if k < 2 else 0, vmax=max(vmax, vmin+1e-30) if k < 2 else max(error[active].max(), 1e-30))
            ax.set(title=label+' / '+('FOM', 'ROM', '|ROM − FOM|')[k], aspect='equal',
                   xlim=limits(coords[active,0]), ylim=limits(coords[active,1]), xlabel='Reference x (m)', ylabel='Reference y (m)')
            fig.colorbar(artist, ax=ax, shrink=.6, label=('m', 'm/s', 'Pa')[j])
    fig.suptitle(f'{title} • final fields, t={times[-1]:g} s')
    fig.savefig(out/'spatial_comparison.png', dpi=150); plt.close(fig)
    metadata = dict(run_dir=str(run), rollout=str(source), frames=len(times),
                    time_range=[float(times[0]),float(times[-1])], fps=args.fps,
                    displacement_scale=1, probe_reference=tip,
                    timing='One saved state per frame; equal playback duration per frame',
                    relative_error_zero_reference='Undefined; written as NaN in CSV/NPZ',
                    num_modes=summary.get('num_modes'))
    if not args.static_only:
        edges = np.sort(np.concatenate([cells[solid][:,[0,1]], cells[solid][:,[1,2]], cells[solid][:,[2,0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        outline = unique[counts == 1]
        xy = (coords+shape)[None,None,:,:]+displacements
        solid_vertices = np.unique(cells[solid])
        solid_xy = xy[:,:,solid_vertices,:]
        sx, sy = limits(solid_xy[:,:,:,0]), limits(solid_xy[:,:,:,1])
        # Focus the flow panels around the solid, with a common physical scale.
        length = max(np.ptp(solid_xy[:,:,:,0]), np.ptp(solid_xy[:,:,:,1]), 1e-9)
        fx = (sx[0]-.4*length, sx[1]+.8*length)
        fy = (sy[0]-.3*length, sy[1]+.3*length)
        norm = Normalize(0, max(float(speeds[:,:,np.unique(cells[fluid])].max()), 1e-30))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=list(axes[0]), label='Fluid speed (m/s)', shrink=.7)
        writer = FFMpegWriter(fps=args.fps, codec='libx264', extra_args=['-pix_fmt','yuv420p'])
        with writer.saving(fig, str(out/'fom_rom_animation.mp4'), dpi=120):
            for i, t in enumerate(times):
                for ax in axes.flat: ax.clear()
                for k, label in enumerate(('FOM', 'ROM')):
                    position = xy[k,i]
                    ax = axes[0,k]
                    tri = mtri.Triangulation(position[:,0], position[:,1], cells, mask=~fluid)
                    ax.tripcolor(tri, speeds[k,i], shading='gouraud', cmap='viridis', norm=norm)
                    ax.add_collection(LineCollection(position[outline], color='#222222', linewidth=.8))
                    ax.set(title=label, aspect='equal', xlim=fx, ylim=fy, xlabel='Deformed x (m)', ylabel='Deformed y (m)')
                    axes[1,0].add_collection(LineCollection(position[outline], color=('tab:blue','tab:orange')[k], linestyle=('-','--')[k], label=label))
                    axes[1,1].plot(times[:i+1], tips[k,:i+1,1], ('-','--')[k], label=label)
                axes[1,0].set(title='Solid outline — actual scale', aspect='equal', xlim=sx, ylim=sy, xlabel='x (m)', ylabel='y (m)')
                axes[1,1].set(title='Vertical tip displacement', xlim=(times[0],times[-1]), ylim=limits(tips[:,:,1]), xlabel='Time (s)', ylabel='Displacement (m)')
                axes[1,0].legend(); axes[1,1].legend(); axes[1,1].grid(alpha=.2)
                fig.suptitle(f'{run.name} • t={t:g} s • actual-scale geometry')
                writer.grab_frame()
                if i in (0, len(times)-1): fig.savefig(out/f'animation_frame_{i:03d}.png', dpi=120)
                if i % 20 == 0: print(f'Rendered {i+1}/{len(times)} frames', flush=True)
        plt.close(fig)
        (out/'animation.html').write_text('<!doctype html><title>FOM / ROM</title><video controls autoplay loop style="width:100%;max-width:1440px" src="fom_rom_animation.mp4"></video>')
    (out/'comparison_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Wrote physical comparisons to {out}', flush=True)


if __name__ == '__main__':
    main()
