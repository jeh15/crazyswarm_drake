from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax import (
    jit,
    jacfwd,
    jacrev,
)

from pydrake.solvers import mathematicalprogram as mp
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.mathematicalprogram import SolverOptions

import pdb


# Container for Failure Probability Regression functions:
class FailureProbabilityNamespace():
    def __init__(self, spline_resolution: int):
        self.spline_resolution = spline_resolution
        self.design_variables = jnp.zeros((spline_resolution+1,))
        self.warmstart = np.zeros((spline_resolution+1,))
        self._ub_tol = 0.99
        self.jit_functions(spline_resolution=spline_resolution)

    # JAX Methods:
    @partial(jit, static_argnums=(0,), static_argnames=['spline_resolution'])
    def _objective_function(
        self,
        y: jax.Array,
        x: jax.Array,
        y_data: jax.Array,
        x_data: jax.Array,
        spline_resolution: int,
    ) -> jnp.ndarray:
        # Format the data:
        x_data = jnp.reshape(x_data, (spline_resolution, -1))
        y_data = jnp.reshape(y_data, (spline_resolution, -1))

        def linear_interpolation(x, y, x_data, y_data):
            output = []
            for i in range(0, x.shape[0]-1):
                output.append(y_data[i, :] - (y[i] + (x_data[i, :] - x[i]) * (y[i+1] - y[i]) / (x[i+1] - x[i])))
            return jnp.array(output)

        minimize_error = linear_interpolation(x, y, x_data, y_data).flatten()
        objective_function = jnp.sum(minimize_error ** 2, axis=0)

        return objective_function

    # JIT Jax Methods
    def jit_functions(self, spline_resolution: int):
        objective_func = lambda y, x, yd, xd: self._objective_function(
            y=y,
            x=x,
            y_data=yd,
            x_data=xd,
            spline_resolution=spline_resolution,
        )

        # Compute H and f matrcies for objective function:
        self.H_function = jax.jit(jacfwd(jacrev(objective_func)))
        self.f_function = jax.jit(jacfwd(objective_func))

    # Configure OSQP:
    def configure_solver(self):
        self.osqp = OsqpSolver()
        self.solver_options = SolverOptions()
        self.solver_options.SetOption(self.osqp.solver_id(), "rho", 1e-02)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_abs", 1e-03)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_rel", 1e-03)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_prim_inf", 1e-03)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_dual_inf", 1e-03)
        self.solver_options.SetOption(self.osqp.solver_id(), "max_iter", 5000)
        self.solver_options.SetOption(self.osqp.solver_id(), "polish", True)
        self.solver_options.SetOption(self.osqp.solver_id(), "polish_refine_iter", 3)
        self.solver_options.SetOption(self.osqp.solver_id(), "warm_start", True)
        self.solver_options.SetOption(self.osqp.solver_id(), "verbose", False)

    # Initialize and Solve Optimization:
    def initialize_optimization(self, data: jax.Array) -> np.ndarray:
        # Create evenly spaced knot points:
        knot_points = np.linspace(
            start=data[0, 0],
            stop=data[0, -1],
            num=self.spline_resolution+1,
        )

        # Update Optimization Matrices:
        H = self.H_function(self.design_variables, knot_points, data[1, :], data[0, :])
        f = self.f_function(self.design_variables, knot_points, data[1, :], data[0, :])

        # Create Handle for Mathematical Program:
        self.prog = mp.MathematicalProgram()

        # Create Handle for Design Variables:
        self.opt_vars = self.prog.NewContinuousVariables(
            rows=self.spline_resolution+1,
            name="y",
        )

        # Add Design Variable Bounds:
        self.prog.AddBoundingBoxConstraint(
            np.zeros(self.opt_vars.shape),
            self._ub_tol * np.ones(self.opt_vars.shape),
            self.opt_vars,
        )

        # Add Objective Function:
        self.objective_function = self.prog.AddQuadraticCost(
            Q=H,
            b=f,
            vars=self.opt_vars,
            is_convex=True,
        )

        # Solve Constrained Least Squares:
        self.solution = self.osqp.Solve(
            self.prog,
            self.warmstart,
            self.solver_options,
        )

        self.run_time = self.solution.get_solver_details().run_time

        if not self.solution.is_success():
            print(f"Optimization did not solve!")
            print(f"Solver Status: {self.solution.get_solver_details().status_val}")
            print(f"Objective Function Convex: {self.objective_function.evaluator().is_convex()}")
            pdb.set_trace()

        self.warmstart = self.solution.GetSolution(self.opt_vars)

        # Return Solution
        return np.vstack(
            [knot_points, self.solution.GetSolution(self.opt_vars)],
        )

    # Update Function for Optimization:
    def update(self,  data: jax.Array) -> np.ndarray:
        # Create evenly spaced knot points:
        knot_points = np.linspace(
            start=data[0, 0],
            stop=data[0, -1],
            num=self.spline_resolution+1,
        )

        # Update Optimization Matrices:
        H = self.H_function(self.design_variables, knot_points, data[1, :], data[0, :])
        f = self.f_function(self.design_variables, knot_points, data[1, :], data[0, :])

        self.objective_function.evaluator().UpdateCoefficients(
            new_Q=H,
            new_b=f,
        )

        # Solve Constrained Least Squares:
        self.solution = self.osqp.Solve(
            self.prog,
            self.warmstart,
            self.solver_options,
        )

        self.run_time = self.solution.get_solver_details().run_time
        
        if not self.solution.is_success():
            print(f"Optimization did not solve!")
            print(f"Solver Status: {self.solution.get_solver_details().status_val}")
            print(f"Objective Function Convex: {self.objective_function.evaluator().is_convex()}")
            pdb.set_trace()

        self.warmstart = self.solution.GetSolution(self.opt_vars)

        # Return knot points for Log-Survival Regression:
        return np.vstack(
            [knot_points, self.solution.GetSolution(self.opt_vars)],
        )
