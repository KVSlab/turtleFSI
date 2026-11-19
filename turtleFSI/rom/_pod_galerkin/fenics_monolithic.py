"""FEniCS adapter for intrusive monolithic POD-Galerkin ROMs.

This module rebuilds the same variational problem setup used by
``turtleFSI.monolithic`` but exposes residual/Jacobian assembly as methods.
That lets the POD-Galerkin reduced Newton solver call:

``R_r(a) = Phi.T R(mean + Phi a)``
``J_r(a) = Phi.T J(mean + Phi a) Phi``

without entering the full-order time loop.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from dolfin import (
    Constant,
    FacetNormal,
    FiniteElement,
    Function,
    FunctionSpace,
    Measure,
    MixedElement,
    TestFunctions,
    TrialFunction,
    VectorElement,
    assemble,
    derivative,
    split,
)

from turtleFSI.problems import default_variables
from turtleFSI.cases.snapshots.dvp_export import active_pressure_dofs, gather_vector_to_root
from turtleFSI.modules.newton_block_forms import build_equation_3_blocks


@dataclass
class ProjectedAssembly:
    residual: np.ndarray
    jacobian: np.ndarray


@dataclass
class ProjectedEquation3Components:
    """Projected Eq. 3 blocks assembled from named Table I component forms."""

    block_names: tuple[str, ...]
    blocks: dict[tuple[str, str], np.ndarray]
    component_blocks: dict[tuple[str, ...], dict[tuple[str, str], np.ndarray]]

    def as_operator(self) -> np.ndarray:
        rows = []
        for row_name in self.block_names:
            rows.append(
                np.hstack([self.blocks[(row_name, column_name)] for column_name in self.block_names])
            )
        return np.vstack(rows)


@dataclass
class Equation3ProjectionCheck:
    """Comparison between aggregate projected Jacobian and Table I block sum."""

    projected_jacobian: np.ndarray
    equation_3_operator: np.ndarray
    difference: np.ndarray
    component_projection: ProjectedEquation3Components

    @property
    def difference_norm(self) -> float:
        return float(np.linalg.norm(self.difference))

    @property
    def jacobian_norm(self) -> float:
        return float(np.linalg.norm(self.projected_jacobian))

    @property
    def relative_difference_norm(self) -> float:
        denominator = self.jacobian_norm
        if denominator == 0.0:
            return 0.0 if self.difference_norm == 0.0 else float("inf")
        return float(self.difference_norm / denominator)


class FEniCSMonolithicProblem:
    """Reusable monolithic FEniCS problem for ROM projection."""

    def __init__(self, namespace: dict[str, Any]):
        self.namespace = namespace
        self.DVP = namespace["DVP"]
        self.dvp_ = namespace["dvp_"]
        self.F = namespace["F"]
        self.J = namespace["J"]
        self.bcs = namespace["bcs"]
        self.compiler_parameters = namespace["compiler_parameters"]

    @classmethod
    def build(
        cls,
        problem_name: str,
        overrides: dict[str, Any] | None = None,
        cwd: Path | None = None,
    ) -> "FEniCSMonolithicProblem":
        namespace = build_monolithic_namespace(
            problem_name=problem_name,
            overrides={} if overrides is None else overrides,
            cwd=Path.cwd() if cwd is None else cwd,
        )
        return cls(namespace)

    @property
    def vector_size(self) -> int:
        return int(self.dvp_["n"].vector().size())

    def set_state(self, state: np.ndarray, time_level: str = "n") -> None:
        self._set_vector(self.dvp_[time_level].vector(), state)

    def set_previous_state(self, state: np.ndarray) -> None:
        self.set_state(state, time_level="n-1")

    def set_previous_previous_state(self, state: np.ndarray) -> None:
        self.set_state(state, time_level="n-2")

    def set_time(self, t: float) -> None:
        self.namespace["t"] = float(t)
        problem_module = self.namespace.get("problem_module")
        if problem_module is not None and hasattr(problem_module, "pre_solve"):
            updates = problem_module.pre_solve(**self.namespace)
            if updates is not None:
                self.namespace.update(updates)

    def get_state(self, time_level: str = "n") -> np.ndarray:
        values = gather_vector_to_root(self.dvp_[time_level].vector())
        if values is None:
            return np.zeros(0, dtype=float)
        return values

    def dirichlet_dofs(self) -> np.ndarray:
        dofs: set[int] = set()
        for bc in self.bcs:
            dofs.update(int(dof) for dof in bc.get_boundary_values().keys())
        return np.asarray(sorted(dofs), dtype=np.int64)

    def free_dofs(self) -> np.ndarray:
        constrained = self.dirichlet_dofs()
        mask = np.ones(self.vector_size, dtype=bool)
        mask[constrained] = False
        return np.flatnonzero(mask).astype(np.int64)

    def fluid_pressure_dofs(self) -> np.ndarray:
        return active_pressure_dofs(
            DVP=self.DVP,
            domains=self.namespace["domains"],
            fluid_domain_ids=self.namespace["dx_f_id_list"],
        )

    def constrain_increment_basis(self, basis: np.ndarray) -> np.ndarray:
        constrained = self.dirichlet_dofs()
        constrained_basis = np.asarray(basis, dtype=float).copy()
        constrained_basis[constrained, :] = 0.0
        return constrained_basis

    def apply_state_bcs(self, state: np.ndarray, time_level: str = "n") -> np.ndarray:
        self.set_state(state, time_level=time_level)
        vector = self.dvp_[time_level].vector()
        for bc in self.bcs:
            bc.apply(vector)
        return self.get_state(time_level)

    def residual(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        rhs = assemble(-self.F, form_compiler_parameters=self.compiler_parameters)
        for bc in self.bcs:
            bc.apply(rhs, self.dvp_["n"].vector())
        values = gather_vector_to_root(rhs)
        if values is None:
            return np.zeros(0, dtype=float)
        return -values

    def assemble_jacobian(self):
        matrix = assemble(
            self.J,
            form_compiler_parameters=self.compiler_parameters,
            keep_diagonal=True,
        )
        matrix.ident_zeros()
        for bc in self.bcs:
            bc.apply(matrix)
        return matrix

    def projected_residual(self, basis: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
        residual = self.residual(state)
        return np.asarray(basis, dtype=float).T @ residual.reshape(-1, 1)

    def projected_jacobian(self, basis: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            self.set_state(state)
        phi = np.asarray(basis, dtype=float)
        matrix = self.assemble_jacobian()
        projected_columns = []
        for column in range(phi.shape[1]):
            product = self._matrix_vector_product(matrix, phi[:, column])
            projected_columns.append(phi.T @ product)
        return np.asarray(projected_columns, dtype=float).T

    def projected_assembly(
        self,
        basis: np.ndarray,
        state: np.ndarray | None = None,
    ) -> ProjectedAssembly:
        if state is not None:
            self.set_state(state)
        residual = self.residual()
        jacobian = self.projected_jacobian(basis)
        return ProjectedAssembly(
            residual=np.asarray(basis, dtype=float).T @ residual.reshape(-1, 1),
            jacobian=jacobian,
        )

    def projected_equation_6(self, block_basis, state=None, constrain_basis=True):
        """Assemble named equation (6) operators and check their block sum.

        Compare physical forms before boundary modifications. Then add the
        separately reported Dirichlet/inactive-row correction so the resulting
        J_N matches the production projected Jacobian for the same basis.
        Reassemble on each call: the operators depend on the current state.
        """
        from turtleFSI.modules.equation_4 import build_equation_4_forms
        from turtleFSI.rom._pod_galerkin.equation_6 import project_equation_6

        if state is not None:
            self.set_state(state)
        forms = build_equation_4_forms(**self.namespace)
        def matrix(form):
            if isinstance(form, (int, float)) and form == 0:
                return None
            return assemble(form, form_compiler_parameters=self.compiler_parameters, keep_diagonal=True)

        operators = {name: matrix(form) for name, form in forms["operators"].items()}
        transform = self.constrain_increment_basis if constrain_basis else np.asarray
        Z_d = transform(block_basis.displacement)
        Z_u_s = transform(block_basis.velocity_enriched)
        Z_p = transform(block_basis.pressure)
        phi = np.hstack([Z_d, Z_u_s, Z_p])
        named = project_equation_6(operators, Z_d, Z_u_s, Z_p, self._project_matrix_between_bases)
        aggregate_raw = self._project_matrix_between_bases(matrix(self.J), phi, phi)
        physical = named.pop("J_N")
        difference = physical - aggregate_raw
        if not np.allclose(physical, aggregate_raw, rtol=1e-8, atol=1e-10):
            raise ValueError(
                "Equation (6) named operators disagree with the full Jacobian: "
                f"difference norm {np.linalg.norm(difference):.6g}. "
                "Check additional loads/supports and equation (4) model compatibility."
            )
        aggregate = self.projected_jacobian(phi)
        boundary_correction = aggregate - aggregate_raw
        named.update(
            equation_6_physical_J_N=physical,
            equation_6_aggregate_raw_J_N=aggregate_raw,
            equation_6_difference_N=difference,
            equation_6_boundary_correction_N=boundary_correction,
            equation_6_aggregate_J_N=aggregate,
            J_N=physical + boundary_correction,
        )
        self.last_equation_6 = named
        return named

    def projected_equation_3_component_blocks(
        self,
        block_basis,
        state: np.ndarray | None = None,
        constrain_basis: bool = True,
    ) -> ProjectedEquation3Components:
        """Project named Table I component forms into Ballarin/Rozza blocks.

        The result is the left-hand side block matrix in Eq. 3, grouped by
        reduced displacement, velocity/supremizer, and pressure spaces. Boundary
        conditions are handled by optionally zeroing constrained rows in the
        supplied embedded bases; Dirichlet identity rows are not assigned to any
        individual physical component.
        """
        if state is not None:
            self.set_state(state)

        named_basis = self._equation_3_named_basis_blocks(block_basis, constrain_basis)
        block_names = tuple(named_basis.keys())
        totals = self._zero_reduced_blocks(named_basis, block_names)
        component_blocks: dict[tuple[str, ...], dict[tuple[str, str], np.ndarray]] = {}

        equation_3_jacobian = self.namespace.get("equation_3_jacobian")
        if equation_3_jacobian is not None and equation_3_jacobian != 0:
            matrix = assemble(
                equation_3_jacobian,
                form_compiler_parameters=self.compiler_parameters,
                keep_diagonal=True,
            )
            projected = {}
            for row_name in block_names:
                row_basis = named_basis[row_name]
                for column_name in block_names:
                    column_basis = named_basis[column_name]
                    block = self._project_matrix_between_bases(matrix, row_basis, column_basis)
                    projected[(row_name, column_name)] = block
                    totals[(row_name, column_name)] += block
            component_blocks[("equation_3_jacobian",)] = projected
            return ProjectedEquation3Components(
                block_names=block_names,
                blocks=totals,
                component_blocks=component_blocks,
            )

        component_tree = self.namespace.get("equation_3_component_blocks", {})
        for path, form in self._iter_equation_3_component_forms(component_tree):
            row_names = self._equation_3_row_names(path, block_names)
            column_names = self._equation_3_column_names(path, block_names)
            if not row_names or not column_names:
                continue
            matrix = assemble(
                form,
                form_compiler_parameters=self.compiler_parameters,
                keep_diagonal=True,
            )
            projected = {}
            for row_name in row_names:
                row_basis = named_basis[row_name]
                for column_name in column_names:
                    column_basis = named_basis[column_name]
                    block = self._project_matrix_between_bases(matrix, row_basis, column_basis)
                    projected[(row_name, column_name)] = block
                    totals[(row_name, column_name)] += block
            component_blocks[path] = projected

        return ProjectedEquation3Components(
            block_names=block_names,
            blocks=totals,
            component_blocks=component_blocks,
        )

    def check_equation_3_projection(
        self,
        block_basis,
        state: np.ndarray | None = None,
        constrain_basis: bool = True,
    ) -> Equation3ProjectionCheck:
        """Compare the summed Table I component blocks with ``Phi.T J Phi``."""
        named_basis = self._equation_3_named_basis_blocks(block_basis, constrain_basis)
        phi = np.hstack([named_basis[name] for name in named_basis])
        projected_jacobian = self.projected_jacobian(phi, state)
        component_projection = self.projected_equation_3_component_blocks(
            block_basis,
            state=None,
            constrain_basis=constrain_basis,
        )
        equation_3_operator = component_projection.as_operator()
        return Equation3ProjectionCheck(
            projected_jacobian=projected_jacobian,
            equation_3_operator=equation_3_operator,
            difference=equation_3_operator - projected_jacobian,
            component_projection=component_projection,
        )

    def _matrix_vector_product(self, matrix, vector_values: np.ndarray) -> np.ndarray:
        template = self.dvp_["n"].vector()
        in_vec = template.copy()
        out_vec = template.copy()
        self._set_vector(in_vec, vector_values)
        out_vec.zero()
        matrix.mult(in_vec, out_vec)
        values = gather_vector_to_root(out_vec)
        if values is None:
            return np.zeros(0, dtype=float)
        return values

    def _project_matrix_between_bases(
        self,
        matrix,
        row_basis: np.ndarray,
        column_basis: np.ndarray,
    ) -> np.ndarray:
        columns = []
        for column in range(column_basis.shape[1]):
            product = self._matrix_vector_product(matrix, column_basis[:, column])
            columns.append(row_basis.T @ product.reshape(-1, 1))
        if not columns:
            return np.zeros((row_basis.shape[1], 0), dtype=float)
        return np.hstack(columns)

    def _equation_3_named_basis_blocks(self, block_basis, constrain_basis: bool) -> dict[str, np.ndarray]:
        blocks = {
            "displacement": block_basis.displacement,
            "velocity": block_basis.velocity,
            "pressure": block_basis.pressure,
        }
        if block_basis.supremizer is not None:
            blocks["supremizer"] = block_basis.supremizer
        ordered = {}
        for name in ("displacement", "velocity", "supremizer", "pressure"):
            basis = blocks.get(name)
            if basis is None or basis.shape[1] == 0:
                continue
            ordered[name] = (
                self.constrain_increment_basis(basis)
                if constrain_basis
                else np.asarray(basis, dtype=float)
            )
        return ordered

    @staticmethod
    def _zero_reduced_blocks(
        named_basis: dict[str, np.ndarray],
        block_names: tuple[str, ...],
    ) -> dict[tuple[str, str], np.ndarray]:
        return {
            (row_name, column_name): np.zeros(
                (named_basis[row_name].shape[1], named_basis[column_name].shape[1]),
                dtype=float,
            )
            for row_name in block_names
            for column_name in block_names
        }

    @classmethod
    def _iter_equation_3_component_forms(cls, tree, path: tuple[str, ...] = ()):
        if isinstance(tree, dict):
            for key, value in tree.items():
                yield from cls._iter_equation_3_component_forms(value, path + (str(key),))
            return
        if tree == 0 or (path and "minus" in path[-1]):
            return
        yield path, tree

    @staticmethod
    def _equation_3_row_names(path: tuple[str, ...], block_names: tuple[str, ...]) -> tuple[str, ...]:
        if len(path) < 2:
            return ()
        section = path[:2]
        if section in (("fluid", "fluid_momentum"), ("solid", "solid_momentum")):
            candidates = ("velocity", "supremizer")
        elif section in (("fluid", "fluid_continuity"),):
            candidates = ("pressure",)
        elif section in (("solid", "extension"), ("extension", "extension_fluid")):
            candidates = ("displacement",)
        else:
            return ()
        return tuple(name for name in candidates if name in block_names)

    @staticmethod
    def _equation_3_column_names(path: tuple[str, ...], block_names: tuple[str, ...]) -> tuple[str, ...]:
        if not path:
            return ()
        name = path[-1]
        if name.startswith("delta_d"):
            candidates = ("displacement",)
        elif name.startswith("delta_u"):
            candidates = ("velocity", "supremizer")
        elif name.startswith("delta_p"):
            candidates = ("pressure",)
        else:
            return ()
        return tuple(block for block in candidates if block in block_names)

    @staticmethod
    def _set_vector(vector, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.shape[0] != vector.size():
            raise ValueError(
                f"state size {array.shape[0]} does not match FEniCS vector size {vector.size()}"
            )
        start, stop = vector.local_range()
        vector.set_local(array[start:stop])
        vector.apply("insert")


def build_monolithic_namespace(
    problem_name: str,
    overrides: dict[str, Any],
    cwd: Path,
) -> dict[str, Any]:
    problem = _import_problem(problem_name, cwd)

    variables = deepcopy(default_variables)
    problem_namespace = {
        key: value for key, value in vars(problem).items() if key != "default_variables"
    }
    variables.update(problem.set_problem_parameters(variables, **problem_namespace))
    variables.update(overrides)
    variables.setdefault("restart_folder", None)

    mesh, domains, boundaries = problem.get_mesh_domain_and_boundaries(**variables)
    variables.update(mesh=mesh, domains=domains, boundaries=boundaries)
    variables["rom_replay"] = True
    if hasattr(problem, "setup_reference_geometry"):
        variables.update(problem.setup_reference_geometry(**variables))

    de = VectorElement("CG", mesh.ufl_cell(), variables["d_deg"])
    ve = VectorElement("CG", mesh.ufl_cell(), variables["v_deg"])
    pe = FiniteElement("CG", mesh.ufl_cell(), variables["p_deg"])
    if variables["extrapolation"] == "biharmonic":
        elem = MixedElement([de, ve, pe, de])
    else:
        elem = MixedElement([de, ve, pe])
    DVP = FunctionSpace(mesh, elem)
    variables.update(DVP=DVP)

    dvp_ = {}
    d_ = {}
    v_ = {}
    p_ = {}
    w_ = {}
    times = ["n-2", "n-1", "n"]
    for time in times:
        dvp = Function(DVP)
        dvp_[time] = dvp
        parts = split(dvp)
        d_[time] = parts[0]
        v_[time] = parts[1]
        p_[time] = parts[2]
        if variables["extrapolation"] == "biharmonic":
            w_[time] = parts[3]
    variables.update(dvp_=dvp_, d_=d_, v_=v_, p_=p_, w_=w_, times=times)

    tests = TestFunctions(DVP)
    variables.update(phi=tests[0], psi=tests[1], gamma=tests[2])
    if variables["extrapolation"] == "biharmonic":
        variables.update(beta=tests[3])

    variables.update(
        k=Constant(variables["dt"]),
        n=FacetNormal(mesh),
        ds=Measure("ds", subdomain_data=boundaries),
        dS=Measure("dS", subdomain_data=boundaries),
        dx=Measure("dx", subdomain_data=domains),
    )

    domain_module = importlib.import_module("turtleFSI.modules.domain")
    variables.update(domain_module.assign_domain_properties(**variables))

    fluid_module = importlib.import_module(f"turtleFSI.modules.{variables['fluid']}")
    solid_module = importlib.import_module(f"turtleFSI.modules.{variables['solid']}")
    extrapolation_module = importlib.import_module(
        f"turtleFSI.modules.{variables['extrapolation']}"
    )
    variables.update(fluid_module.fluid_setup(**variables))
    variables.update(solid_module.solid_setup(**variables))
    variables.update(extrapolation_module.extrapolate_setup(**variables))

    if hasattr(problem, "initiate"):
        initiated = problem.initiate(**variables)
        if initiated is not None:
            variables.update(initiated)
    variables.update(problem.create_bcs(**variables))

    F_lin = variables["F_fluid_linear"] + variables["F_solid_linear"]
    F_nonlin = variables["F_fluid_nonlinear"] + variables["F_solid_nonlinear"]
    F = F_lin + F_nonlin
    chi = TrialFunction(DVP)
    J = derivative(F, dvp_["n"], chi)
    equation_3_namespace = dict(variables)
    equation_3_namespace.pop("DVP", None)
    equation_3_namespace.pop("dvp_", None)
    variables.update(build_equation_3_blocks(DVP=DVP, dvp_=dvp_, **equation_3_namespace))
    variables.update(F=F, J=J, chi=chi, problem=problem_name, problem_module=problem)

    return variables


def _import_problem(problem_name: str, cwd: Path):
    local_problem = cwd / f"{problem_name}.py"
    if local_problem.is_file():
        spec = importlib.util.spec_from_file_location(problem_name, local_problem)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load local problem {local_problem}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(f"turtleFSI.problems.{problem_name}")
