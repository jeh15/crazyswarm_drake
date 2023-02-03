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


class LogSurvivalNamespace():
    def __init__(self, spline_resolution: int):
        self.spline_resolution = spline_resolution
        self.design_variables = jnp.zeros((spline_resolution+1,))
        self.weights = jnp.ones((spline_resolution+1,))
        self.warmstart = np.zeros((spline_resolution+1,))
        self.jit_functions()

    # Jax Methods:
    @partial(jit, static_argnums=(0,))
    def _inequality_constraints(
        self,
        y: jax.Array,
        x: jax.Array,
    ) -> jnp.ndarray:
        m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        inequality_constraints = m[1:] - m[:-1]
        return inequality_constraints

    @partial(jit, static_argnums=(0,))
    def _objective_function(
        self,
        y: jax.Array,
        y_data: jax.Array,
        w: jax.Array,
    ) -> jnp.ndarray:
        minimize_error = w * (y_data - y) ** 2
        objective_function = jnp.sum(minimize_error, axis=0)

        return objective_function

    # JIT Jax Methods:
    def jit_functions(self):
        # Isolate Functions with Lambda Expressions
        inequality_func = lambda y, x: self._inequality_constraints(
            y=y,
            x=x,
        )
        objective_func = lambda y, yd, w: self._objective_function(
            y=y,
            y_data=yd,
            w=w,
        )

        # Compute A and b matrices for inequality constraints:
        self.A_function = jax.jit(jacfwd(inequality_func))
        self.b_function = inequality_func

        # Compute H and f matrices for objective function:
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

    def initialize_optimization(self, data: jax.Array) -> jnp.ndarray:
        # Update Optimization Matrices:
        A = self.A_function(self.design_variables, data[0, :])
        ub = -self.b_function(self.design_variables, data[0, :])
        lb = np.NINF * np.ones(ub.shape)
        H = self.H_function(self.design_variables, data[1, :], self.weights)
        f = self.f_function(self.design_variables, data[1, :], self.weights)

        # Create Handle for Mathematical Program:
        self.prog = mp.MathematicalProgram()

        # Create Handle for Design Variables:
        self.opt_vars = self.prog.NewContinuousVariables(
            rows=self.spline_resolution+1,
            name="y"
        )

        # Add Design Variable Bounds:
        self.prog.AddBoundingBoxConstraint(
            np.NINF * np.ones(self.opt_vars.shape),
            np.zeros(self.opt_vars.shape),
            self.opt_vars,
        )

        # Add Convexity Constraint
        self.inequality_constraints = self.prog.AddLinearConstraint(
            A=A,
            lb=lb,
            ub=ub,
            vars=self.opt_vars,
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

        if not self.solution.is_success():
            print(f"Optimization did not solve!")
            print(f"Solver Status: {self.solution.get_solver_details().status_val}")
            print(f"Objective Function Convex: {self.objective_function.evaluator().is_convex()}")
            pdb.set_trace()

        self.warmstart = self.solution.GetSolution(self.opt_vars)

        # Return knot points for Linear Constraints:
        return jnp.vstack(
            [data[0, :], self.solution.GetSolution(self.opt_vars)],
        )

    # Update Function for Optimization:
    def update(self, data: jax.Array) -> jnp.ndarray:
        # Update Optimization Matrices:
        A = self.A_function(self.design_variables, data[0, :])
        ub = -self.b_function(self.design_variables, data[0, :])
        lb = np.NINF * np.ones(ub.shape)
        H = self.H_function(self.design_variables, data[1, :], self.weights)
        f = self.f_function(self.design_variables, data[1, :], self.weights)

        self.inequality_constraints.evaluator().UpdateCoefficients(
            new_A=A,
            new_lb=lb,
            new_ub=ub,
        )

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

        if not self.solution.is_success():
            print(f"Optimization did not solve!")
            print(f"Solver Status: {self.solution.get_solver_details().status_val}")
            print(f"Objective Function Convex: {self.objective_function.evaluator().is_convex()}")
            pdb.set_trace()

        self.warmstart = self.solution.GetSolution(self.opt_vars)

        # Return Solution:
        return jnp.vstack(
            [data[0, :], self.solution.GetSolution(self.opt_vars)],
        )


if __name__ == "__main__":
    dummy_input = 4
    lsn = LogSurvivalNamespace(spline_resolution=dummy_input)