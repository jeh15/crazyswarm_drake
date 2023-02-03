import numpy as np
import ml_collections

import jax
import jax.numpy as jnp

from pydrake.solvers import mathematicalprogram as mp
from pydrake.common.value import Value
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.mathematicalprogram import SolverOptions
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

# Import Optimization Functions:
import failure_probability_fit as fp
import log_survival_fit as ls

import pdb


class Adversary(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = config.crazyswarm_rate  
        self._OUTPUT_UPDATE_RATE = config.motion_planner_rate
        self._spline_resolution = config.spline_resolution
        self._bin_resolution = config.bin_resolution

        # Make sure number of bins and splines are compatable:
        if (self._bin_resolution < self._spline_resolution):
            print(f"Defaulting bin_resolution == spline_resolution... \nSuggesting that bin_resolution >> spline_resolution.")
            self._bin_resolution = self._spline_resolution
        self._bin_resolution = self._bin_resolution - self._bin_resolution % self._spline_resolution
        if (self._bin_resolution != config.bin_resolution):
            print(f"Changed bin_resolution to {self._bin_resolution} to be compatable with spline_resolution.")

        # Optimization Parameters:
        self._warmstart_fp = jnp.zeros((self._spline_resolution+1,))
        self._warmstart_ls = jnp.zeros((self._spline_resolution+1,))
        self._design_variables = jnp.zeros((self._spline_resolution+1,))
        self._weights = jnp.ones((self._spline_resolution,))

        # Initialize Vector:
        self._data = []

        """ Initialize Abstract States: (Output Only) """
        output_size = np.zeros((self._spline_resolution, 2))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        # Declare Output: Updated Risk Constraint
        self.DeclareVectorOutputPort(
            "risk_constraint_output",
            output_size.shape,
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        """ Declare Input: """
        # Full State From CrazySwarm: (x, y, z, dx, dy, dz, ddx, ddy, ddz)
        self.agent_input = self.DeclareVectorInputPort(
            "agent_position",
            9,
        ).get_index()

        # Current Adversary States: (x, y, z, dx, dy, dz)
        self.adversary_input = self.DeclareVectorInputPort(
            "obstacle_states",
            6,
        ).get_index()

        # Declare Initialization Event to Build Optimizations:
        def on_initialize(context, event):
            self.compile_functions()
            self.configure_solver()
            initialize_data = self.evaluate(context)
            self._data = initialize_data
            data = self.bin_data()
            fp_solution = self.build_failure_probability(data=data)
            ls_solution = self.build_log_survival(data=fp_solution)
            self.constraints = self.update_constraints(data=ls_solution)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Output Update Event:
        def periodic_output_event(context, event):
            current_data = self.evaluate(context)
            self.record_data(current_data)
            data = self.bin_data()
            fp_solution = self.update_failure_probability(data=data)
            ls_solution = self.update_log_survival(data=fp_solution)
            self.constraints = self.update_constraints(data=ls_solution)

        self.DeclarePeriodicEvent(
            period_sec=self._OUTPUT_UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_output_event,
                )
            )

    # Output Port Callback:
    def output_callback(self, context, output):
        pass

    # Configure OSQP:
    def configure_solver(self):
        self.osqp = OsqpSolver()
        self.solver_options = SolverOptions()
        self.solver_options.SetOption(self.osqp.solver_id(), "rho", 1e-04)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_abs", 1e-05)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_rel", 1e-05)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_prim_inf", 1e-05)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_dual_inf", 1e-05)
        self.solver_options.SetOption(self.osqp.solver_id(), "max_iter", 5000)
        self.solver_options.SetOption(self.osqp.solver_id(), "polish", True)
        self.solver_options.SetOption(self.osqp.solver_id(), "polish_refine_iter", 3)
        self.solver_options.SetOption(self.osqp.solver_id(), "warm_start", True)
        self.solver_options.SetOption(self.osqp.solver_id(), "verbose", False)

    # Initialize Constrained Least Squares Optimization:
    def build_failure_probability(self, data: jax.Array) -> jnp.ndarray:
        # Create evenly spaced knot points:
        knot_points = jnp.linspace(
            start=data[0, 0],
            stop=data[0, -1],
            num=self._spline_resolution+1
        )

        # Update Optimization Matrices:
        H = self._H_fp(self._design_variables, knot_points, data[1, :], data[0, :])
        f = self._f_fp(self._design_variables, knot_points, data[1, :], data[0, :])

        # Create Handle for Mathematical Program:
        self.prog_fp = mp.MathematicalProgram()

        # Create Handle for Design Variables:
        self.opt_vars_fp = self.prog_fp.NewContinuousVariables(self._spline_resolution+1, "y")

        # Add Design Variable Bounds:
        self.prog_fp.AddBoundingBoxConstraint(
            np.zeros(self.opt_vars_fp.shape),
            np.ones(self.opt_vars_fp.shape),
            self.opt_vars_fp,
        )

        # Add Objective Function:
        self.objective_function_fp = self.prog.AddQuadraticCost(
            Q=H,
            b=f,
            vars=self.opt_vars_fp,
            is_convex=True,
        )

        # Solve Constrained Least Squares:
        self.solution_fp = self.osqp.Solve(
            self.prog_fp,
            self._warmstart_fp,
            self.solver_options,
        )
        self._warmstart_fp = self.solution_fp.GetSolution(self.opt_vars_fp)

        # Return knot points for Log-Survival Regression:
        return jnp.vstack(
            [knot_points, self.solution_fp.GetSolution(self.opt_vars_fp)],
        )

    def build_log_survival(self, data: jax.Array) -> jnp.ndarray:
        # Update Optimization Matrices:
        A = self._A_ls(self._design_variables, data[0, :])
        ub = -self._b_ls(self._design_variables, data[0, :])
        lb = np.NINF * np.ones(ub.shape)
        H = self._H_ls(self._design_variables, data[1, :], self._weights)
        f = self._f_ls(self._design_variables, data[1, :], self._weights)

        # Create Handle for Mathematical Program:
        self.prog_ls = mp.MathematicalProgram()

        # Create Handle for Design Variables:
        self.opt_vars_ls = self.prog_ls.NewContinuousVariables(self._spline_resolution+1, "y")

        # Add Design Variable Bounds:
        self.prog_ls.AddBoundingBoxConstraint(
            np.NINF * np.ones(self.opt_vars_ls.shape),
            np.zeros(self.opt_vars_ls.shape),
            self.opt_vars_ls,
        )

        # Add Convexity Constraint
        self.inequality_constraints_ls = self.prog_ls.AddLinearConstraint(
            A=A,
            lb=lb,
            ub=ub,
            vars=self.opt_vars_ls,
        )

        # Add Objective Function:
        self.objective_function_ls = self.prog_ls.AddQuadraticCost(
            Q=H,
            b=f,
            vars=self.opt_vars_ls,
            is_convex=True,
        )

        # Solve Constrained Least Squares:
        self.solution_ls = self.osqp.Solve(
            self.prog_ls,
            self._warmstart_ls,
            self.solver_options,
        )
        self._warmstart_ls = self.solution_ls.GetSolution(self.opt_vars_ls)

        # Return knot points for Linear Constraints:
        return jnp.vstack(
            [data[0, :], self.solution_ls.GetSolution(self.opt_vars_ls)],
        )

    # Evaluate Data:
    def evaluate(self, context) -> jnp.ndarray:
        # Get Input Port Values and Convert to Jax Array:
        agent_states = jnp.asarray(self.get_input_port(self.agent_input).Eval(context)[:3])
        adversary_states = jnp.asarray(self.get_input_port(self.adversary_input).Eval(context)[:3])

        # Evaluate if agent has failed:
        distance = jnp.sqrt(
            jnp.sum((agent_states - adversary_states) ** 2, axis=0)
        ) - self._failure_radius

        # Create Data Point Pairs: 
        if distance <= 0:
            x = distance
            y = 1.0
            self._failure_flag = True
        else:
            x = distance
            y = 0.0
            self._failure_flag = False

        return jnp.array([[x], [y]], dtype=float)

    # Record Data:
    def record_data(self, data: jax.Array) -> None:
        vector = jnp.append(self._data, data, axis=1)
        indx = jnp.argsort(vector[0, :])
        self._data = vector[:, indx]

    # Bin Values:
    def bin_data(self) -> jnp.ndarray:
        # Binning Operation
        sums, edges = jnp.histogram(
            self._data[0, :],
            bins=self._bin_resolution,
            weights=self._data[1, :],
        )
        counts, _ = jnp.histogram(
            self._data[0, :],
            bins=self._bin_resolution,
        )
        y = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums, dtype=float),
            where=counts!=0,
        )
        x = (edges[:-1] + edges[1:]) / 2
        binned_data = jnp.vstack([x, y])

        return binned_data

    # Failure Probability Fit:
    def update_failure_probability(self, data: jax.Array) -> jnp.ndarray:
        # Create evenly spaced knot points:
        knot_points = jnp.linspace(
            start=data[0, 0],
            stop=data[0, -1],
            num=self._spline_resolution+1
        )

        # Update Optimization Matrices:
        H = self._H_fp(self._design_variables, knot_points, data[1, :], data[0, :])
        f = self._f_fp(self._design_variables, knot_points, data[1, :], data[0, :])

        self.objective_function_fp.evaluator().UpdateCoefficients(
            new_Q=H,
            new_b=f,
        )

        # Solve Constrained Least Squares:
        self.solution_fp = self.osqp.Solve(
            self.prog_fp,
            self._warmstart_fp,
            self.solver_options,
        )
        self._warmstart_fp = self.solution_fp.GetSolution(self.opt_vars_fp)

        # Return knot points for Log-Survival Regression:
        return jnp.vstack(
            [knot_points, self.solution_fp.GetSolution(self.opt_vars_fp)],
        )

    # Log-Survival Fit:
    def update_log_survival(self, data: jax.Array) -> jnp.ndarray:
        # Update Optimization Matrices:
        A = self._A_ls(self._design_variables, data[0, :])
        ub = -self._b_ls(self._design_variables, data[0, :])
        lb = np.NINF * np.ones(ub.shape)
        H = self._H_ls(self._design_variables, data[1, :], self._weights)
        f = self._f_ls(self._design_variables, data[1, :], self._weights)

        self.inequality_constraints_ls.evaluator().UpdateCoefficients(
            new_A=A,
            new_lb=lb,
            new_ub=ub,
        )

        self.objective_function_ls.evaluator().UpdateCoefficients(
            new_Q=H,
            new_b=f,
        )

        # Solve Constrained Least Squares:
        self.solution_ls = self.osqp.Solve(
            self.prog_ls,
            self._warmstart_ls,
            self.solver_options,
        )
        self._warmstart_ls = self.solution_ls.GetSolution(self.opt_vars_ls)

        # Return knot points for Linear Constraints:
        return jnp.vstack(
            [data[0, :], self.solution_ls.GetSolution(self.opt_vars_ls)],
        )

    # Construct Risk Constraint:
    def update_constraints(self, data: jax.Array) -> jnp.ndarray:
        poly_coefficients = []
        for i in range(0, self._spline_resolution):
            p = np.polyfit(data[0, i:i+2], data[1, i:i+2], 1)
            poly_coefficients.append(p)
        return jnp.asarray(poly_coefficients)

    # Compile JIT functions and create function handles:
    def compile_functions(self):
        self._H_fp, self._f_fp = fp.jit_functions(num_spline=self._spline_resolution)
        self._A_ls, self._b_ls, self._H_ls, self._f_ls = ls.jit_functions()
