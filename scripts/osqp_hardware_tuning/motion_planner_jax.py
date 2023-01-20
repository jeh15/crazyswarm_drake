from functools import partial
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import ml_collections

from jax import (
    grad,
    jit,
    jacfwd,
    jacrev
)

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

import pdb


class QuadraticProgram(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Class Parameters:
        self._UPDATE_RATE = config.motion_planner_rate

        # Mathematical Program Parameters: Takes from Config
        self._nodes = config.nodes
        self._time_horizon = config.time_horizon
        self._state_dimension = config.state_dimension
        self._dt = config.dt
        self._mass = 0.027 # Actual Crazyflie Mass
        self._friction = 0.01

        # State Size for Optimization: (Seems specific to this implementation should not be a config param)
        self._state_size = self._state_dimension * 2
        self._full_size = self._state_dimension * 3
        self._setpoint = jnp.zeros(
            (self._full_size * self._nodes + self._full_size * (self._nodes - 1),),
            dtype=float,
        )
        self._state_bounds = jnp.asarray(
            [5, 10, 0.1],
            dtype=float,
        )
        self._weights = jnp.asarray(
            [1.0, 0.5],
            dtype=float,
        )

        # Initialize for T = 0:
        self._full_state_trajectory = np.zeros(
            (self._full_size * self._nodes,),
            dtype=float,
        )

        self._warm_start = np.zeros(
            (self._full_size * self._nodes + self._full_size * (self._nodes - 1),),
            dtype=float,
        )

        # Initialize Abstract States:
        # Output: (x, y, dx, dy, ux, uy)
        output_size = np.zeros((self._full_size * self._nodes,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        """ Declare Input: """
        # Full State From CrazySwarm: (x, y, z, dx, dy, dz, ddx, ddy, ddz)
        self.initial_condition_input = self.DeclareVectorInputPort(
            "initial_condition",
            9,
        ).get_index()

        self.target_input = self.DeclareVectorInputPort(
            "target_position",
            self._state_dimension,
        ).get_index()

        """ Declare Output: """
        self.DeclareVectorOutputPort(
            "motion_plan_output",
            np.size(output_size),
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        # Declare Initialization Event:
        def on_initialize(context, event):
            self.build_optimization(context, event)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
            )
        )

        # Declare Update Event: Solve Quadratic Program
        def on_periodic(context, event):
            self.update_qp(context, event)

        self.DeclarePeriodicEvent(
            period_sec=self._UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=on_periodic,
                )
            )

    # Output Port Callback:
    def output_callback(self, context, output_motion_plan):
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_value = a_state.get_mutable_value()
        output_motion_plan.SetFromVector(a_value.get_mutable_value())

    # Jax Methods:
    @partial(jit, static_argnums=(0,), static_argnames=['mass', 'friction', 'dt', 'num_states', 'num_nodes'])
    def _equality_constraints(self, q: jax.Array, initial_conditions: jax.Array, mass: float, friction: float, dt: float, num_states: int, num_nodes: int) -> jnp.ndarray:
        print(f"COMPILED EQUALITY CONSTRAINTS")
        """
        Helper Functions:
            1. Hermite-Simpson Collocation
        """
        # 1. Hermite-Simpson Collocation  :      
        def _collocation_constraint(x: jax.Array, dt: float) -> jnp.ndarray:
            collocation = x[0][1:] - x[0][:-1] - (1.0 / 6.0) * dt * (x[1][:-1] + 4.0 * x[3][:] + x[1][1:])
            midpoint = x[2][:] - (1.0 / 2.0) * (x[0][:-1] + x[0][1:]) + (1.0 / 8.0) * dt * (x[1][:-1] - x[1][1:])
            return jnp.concatenate([collocation, midpoint], axis=0)

        """
        Equality Constraints:
            1. Initial Condition
            2. Collocation Constraint
        """

        # Sort State Vector:
        q_m = q[(num_states * num_nodes):]
        q = q[:(num_states * num_nodes)]

        q = q.reshape((num_states, -1))
        q_m = q_m.reshape((num_states, -1))

        # State Nodes:
        x = q[0, :]
        y = q[1, :]
        dx = q[2, :]
        dy = q[3, :]
        ux = q[4, :]
        uy = q[5, :]

        # State Mid Points:
        x_m = q_m[0, :]
        y_m = q_m[1, :]
        dx_m = q_m[2, :]
        dy_m = q_m[3, :]
        ux_m = q_m[4, :]
        uy_m = q_m[5, :]
       
        # 1. Initial Condition Constraints:
        initial_condition = jnp.array([
            x[0] - initial_conditions[0],
            y[0] - initial_conditions[1],
            dx[0] - initial_conditions[3],
            dy[0] - initial_conditions[4],
            ux[0] - initial_conditions[6],
            uy[0] - initial_conditions[7],
        ], dtype=float)
        
        # 2. Collocation Constraints:
        ddx, ddx_m = (ux - friction * dx) / mass, (ux_m - friction * dx_m) / mass
        ddy, ddy_m = (uy - friction * dy) / mass, (uy_m - friction * dy_m) / mass
        x_defect = _collocation_constraint([x, dx, x_m, dx_m], dt)
        y_defect = _collocation_constraint([y, dy, y_m, dy_m], dt)
        dx_defect = _collocation_constraint([dx, ddx, dx_m, ddx_m], dt)
        dy_defect = _collocation_constraint([dy, ddy, dy_m, ddy_m], dt)

        equality_constraint = jnp.concatenate(
            [
                initial_condition,
                x_defect,
                y_defect,
                dx_defect,
                dy_defect,
            ]
        )

        return equality_constraint


    @partial(jit, static_argnums=(0,), static_argnames=['num_states', 'num_nodes'])
    def _inequality_constraints(self, q: jax.Array, state_bounds: jax.Array, num_states: int, num_nodes: int) -> jnp.ndarray:
        print(f"COMPILED INEQUALITY CONSTRAINTS")
        """
        Inquality Constraints:
            1. State Bounds
        """

        # Sort State Vector:
        q_m = q[(num_states * num_nodes):]
        q = q[:(num_states * num_nodes)]

        q = q.reshape((num_states, -1))
        q_m = q_m.reshape((num_states, -1))

        # State Nodes:
        x = q[0, :]
        y = q[1, :]
        dx = q[2, :]
        dy = q[3, :]
        ux = q[4, :]
        uy = q[5, :]

        # State Mid Points:
        x_m = q_m[0, :]
        y_m = q_m[1, :]
        dx_m = q_m[2, :]
        dy_m = q_m[3, :]
        ux_m = q_m[4, :]
        uy_m = q_m[5, :]

        bounds = jnp.concatenate(
            [
                x - state_bounds[0], x_m - state_bounds[0],
                y - state_bounds[0], y_m - state_bounds[0],
                dx - state_bounds[1], dx_m - state_bounds[1],
                dy - state_bounds[1], dy_m - state_bounds[1],
                ux - state_bounds[2], ux_m - state_bounds[2],
                uy - state_bounds[2], uy_m - state_bounds[2],
            ],
            axis=0
        )

        return bounds

    @partial(jit, static_argnums=(0,), static_argnames=['dt', 'num_states', 'num_nodes'])
    def _objective_function(self, q: jax.Array, target_position: jax.Array, w: jax.Array, dt: float, num_states: int, num_nodes: int) -> jnp.ndarray:
        print(f"COMPILED OBJECTIVE FUNCTION")
        """
        Helper Functions:
            1. Calculate Error
            2. Hermite-Simpson Objective Function Form
        """

        # Helper Functions:
        def _calc_error(x: jax.Array, target_position: jax.Array) -> jnp.ndarray:
            return x - target_position

        def _hermite_simpson_objective(x: jax.Array, w: float, dt: float) -> jnp.ndarray:
            return w * (dt / 6.0) * jnp.sum(x[0][:-1] ** 2 + 4 * (x[1][:] ** 2) + x[0][1:] ** 2, axis=0)
        
        """
        Objective Function:
            1. State Error Objective
            2. Control Effort Objective
        """

        # Sort State Vector:
        q_m = q[(num_states * num_nodes):]
        q = q[:(num_states * num_nodes)]

        q = q.reshape((num_states, -1))
        q_m = q_m.reshape((num_states, -1))

        # State Nodes:
        x = q[0, :]
        y = q[1, :]
        ux = q[4, :]
        uy = q[5, :]

        # State Mid Points:
        x_m = q_m[0, :]
        y_m = q_m[1, :]
        ux_m = q_m[4, :]
        uy_m = q_m[5, :]

        minimize_error_x = _hermite_simpson_objective(
            x=[
                _calc_error(x=x[:], target_position=target_position[0]),
                _calc_error(x=x_m[:], target_position=target_position[0]),
            ],
            w=w[0],
            dt=dt,
        )

        minimize_error_y = _hermite_simpson_objective(
            x=[
                _calc_error(x=y[:], target_position=target_position[1]),
                _calc_error(x=y_m[:], target_position=target_position[1]),
            ],
            w=w[0],
            dt=dt,
        )

        minimize_effort_x = _hermite_simpson_objective(
            x=[ux, ux_m],
            w=w[1],
            dt=dt,
        )

        minimize_effort_y = _hermite_simpson_objective(
            x=[uy, uy_m],
            w=w[1],
            dt=dt,
        )

        minimize_error = minimize_error_x + minimize_error_y
        minimize_effort = minimize_effort_x + minimize_effort_y

        objective_function = minimize_error + minimize_effort

        return objective_function

    def build_optimization(self, context, event):
        # Get Input Port Values and Convert to Jax Array:
        initial_conditions = self.get_input_port(self.initial_condition_input).Eval(context)
        # Conver Acceleration to Control Input:
        initial_conditions[-3] = self.compute_control(
            ddq=initial_conditions[-3],
            dq=initial_conditions[3],
            limit=self._state_bounds[2].astype(float),
        )
        initial_conditions[-2] = self.compute_control(
            ddq=initial_conditions[-2],
            dq=initial_conditions[4],
            limit=self._state_bounds[2].astype(float),
        )
        initial_conditions = jnp.asarray(initial_conditions)
        target_positions = jnp.asarray(self.get_input_port(self.target_input).Eval(context))


        # Isolate Functions to Lambda Functions and wrap them in staticmethod:
        self.equality_func = lambda x, ic: self._equality_constraints(
            q=x,
            initial_conditions=ic,
            mass=self._mass,
            friction=self._friction,
            dt=self._dt,
            num_states=self._full_size,
            num_nodes=self._nodes,
        )

        self.inequality_func = lambda x: self._inequality_constraints(
            q=x,
            state_bounds=self._state_bounds,
            num_states=self._full_size,
            num_nodes=self._nodes,
        )

        self.objective_func = lambda x, qd: self._objective_function(
            q=x,
            target_position=qd,
            w=self._weights,
            dt=self._dt,
            num_states=self._full_size,
            num_nodes=self._nodes,
        )

        # Compute A and b matricies for equality constraints:
        self._A_eq_fn = jax.jit(jacfwd(self.equality_func))
        A_eq = self._A_eq_fn(self._setpoint, initial_conditions)
        b_eq = -self.equality_func(self._setpoint, initial_conditions)

        # Compute A and b matricies for inequality constraints:
        self._A_ineq_fn = jax.jit(jacfwd(self.inequality_func))
        A_ineq = self._A_ineq_fn(self._setpoint)
        b_ineq = -self.inequality_func(self._setpoint)
        
        # Compute H and f matrcies for objective function:
        self._H_fn = jax.jit(jacfwd(jacrev(self.objective_func)))
        self._f_fn = jax.jit(jacfwd(self.objective_func))
        H = self._H_fn(self._setpoint, target_positions)
        f = self._f_fn(self._setpoint, target_positions)

        # Construct OSQP Problem:
        self.prog = mp.MathematicalProgram()

        # State and Control Input Variables:
        self._opt_x = self.prog.NewContinuousVariables(self._full_size * self._nodes, "x")

        # Mid Point Decision Variables for Hermite-Simpson:
        self._opt_x_m = self.prog.NewContinuousVariables(self._full_size * (self._nodes - 1), "x_m")

        # Concatenate Optimaztion Variables:
        opt_vars = np.concatenate([self._opt_x, self._opt_x_m], axis=0)

        # Add Constraints and Objective Function:
        self.equality_constraint = self.prog.AddLinearConstraint(
            A=A_eq,
            lb=b_eq,
            ub=b_eq,
            vars=opt_vars,
        )

        self.inequality_constraint = self.prog.AddLinearConstraint(
            A=A_ineq,
            lb=-b_ineq,
            ub=b_ineq,
            vars=opt_vars,
        )

        self.objective_function = self.prog.AddQuadraticCost(
            Q=H,
            b=f,
            vars=opt_vars,
            is_convex=True,
        )

        # Solve the program:
        """OSQP:"""
        self.osqp = OsqpSolver()
        self.solver_options = SolverOptions()
        self.solver_options.SetOption(self.osqp.solver_id(), "rho", 1e-03)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_abs", 1e-06)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_rel", 1e-06)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_prim_inf", 1e-06)
        self.solver_options.SetOption(self.osqp.solver_id(), "eps_dual_inf", 1e-06)
        self.solver_options.SetOption(self.osqp.solver_id(), "max_iter", 3000)
        self.solver_options.SetOption(self.osqp.solver_id(), "polish", True)
        self.solver_options.SetOption(self.osqp.solver_id(), "polish_refine_iter", 10)
        self.solver_options.SetOption(self.osqp.solver_id(), "warm_start", True)
        self.solver_options.SetOption(self.osqp.solver_id(), "verbose", False)
        self.solution = self.osqp.Solve(
            self.prog,
            self._warm_start,
            self.solver_options,
        )

        # Parse Solution:
        self.parse_solution(context, event)

    def update_qp(self, context, event):
        # Get Input Port Values and Convert to Jax Array:
        initial_conditions = self.get_input_port(self.initial_condition_input).Eval(context)
        # Conver Acceleration to Control Input:
        initial_conditions[-3] = self.compute_control(
            ddq=initial_conditions[-3],
            dq=initial_conditions[3],
            limit=self._state_bounds[2].astype(float),
        )
        initial_conditions[-2] = self.compute_control(
            ddq=initial_conditions[-2],
            dq=initial_conditions[4],
            limit=self._state_bounds[2].astype(float),
        )
        initial_conditions = jnp.asarray(initial_conditions)
        target_positions = jnp.asarray(self.get_input_port(self.target_input).Eval(context))

        # Compute A and b matricies for equality constraints:
        A_eq = self._A_eq_fn(self._setpoint, initial_conditions)
        b_eq = -self.equality_func(self._setpoint, initial_conditions)

        # Compute A and b matricies for inequality constraints:
        A_ineq = self._A_ineq_fn(self._setpoint)
        b_ineq = -self.inequality_func(self._setpoint)
        
        # Compute H and f matrcies for objective function:
        H = self._H_fn(self._setpoint, target_positions)
        f = self._f_fn(self._setpoint, target_positions)

        # Update Constraint and Objective Function:
        self.equality_constraint.evaluator().UpdateCoefficients(
            new_A=A_eq, 
            new_lb=b_eq,
            new_ub=b_eq,
        )

        self.inequality_constraint.evaluator().UpdateCoefficients(
            new_A=A_ineq,
            new_lb=-b_ineq,
            new_ub=b_ineq,
        )

        self.objective_function.evaluator().UpdateCoefficients(
            new_Q=H,
            new_b=f,
        )

        # Solve Updated Optimization:
        self.solution = self.osqp.Solve(
            self.prog,
            self._warm_start,
            self.solver_options,
        )

        # Parse Solution:
        self.parse_solution(context, event)

    # Helper Functions:
    def compute_control(self, ddq: float, dq: float, limit: float) -> float:
        def convert_to_control(self, ddq: float, dq: float) -> float:
            return (ddq * self._mass + self._friction * dq)
        u = convert_to_control(self, ddq, dq)
        return (u if jnp.abs(u) < limit else jnp.sign(u) * limit)

    def compute_acceleration(self, u: np.ndarray, dx: np.ndarray) -> np.ndarray:
        return np.asarray([(u[:] - self._friction * dx[:]) / self._mass]).flatten()

    def parse_solution(self, context, event):
        if not self.solution.is_success():
            print(f"Optimization did not solve!")
            print(f"Solver Status: {self.solution.get_solver_details().status_val}")
            print(f"Objective Function Convex: {self.objective_function.evaluator().is_convex()}")
            pdb.set_trace()

        x_sol = np.reshape(self.solution.GetSolution(self._opt_x), (self._full_size, -1))
        x_m_sol = np.reshape(self.solution.GetSolution(self._opt_x_m), (self._full_size, -1))

        ddx = self.compute_acceleration(x_sol[-2, :], x_sol[2, :])
        ddy = self.compute_acceleration(x_sol[-1, :], x_sol[3, :])

        self._full_state_trajectory = np.concatenate(
            [
                x_sol[0, :], x_sol[1, :],
                x_sol[2, :], x_sol[3, :],
                ddx, ddy
            ],
            axis=0,
        )

        self._warm_start = np.concatenate(
            [
                x_sol[0, :], x_sol[1, :],
                x_sol[2, :], x_sol[3, :],
                x_sol[4, :], x_sol[5, :],
                x_m_sol[0, :], x_m_sol[1, :],
                x_m_sol[2, :], x_m_sol[3, :],
                x_m_sol[4, :], x_m_sol[5, :],
            ], 
            axis=0,
        )

        # How can I clean this up?
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_state.set_value(self._full_state_trajectory)


# Test Instantiation:
if __name__ == "__main__":
    unit_test = QuadraticProgram()
