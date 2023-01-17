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

from pydrake.common.value import Value
from pydrake.all import (
    MathematicalProgram,
    SolverOptions,
    OsqpSolver
)
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
            (self._full_size * self._nodes + self._state_size * (self._nodes - 1),),
            dtype=float,
        )
        self._state_bounds = jnp.asarray(
            [5, 10, 0.5],
            dtype=float,
        )
        self._weights = jnp.asarray(
            [1.0, 0.0],
            dtype=float,
        )

        # Initialize for T = 0:
        self._full_state_trajectory = np.zeros(
            (self._full_size * self._nodes,),
            dtype=float,
        )

        self._warm_start = np.zeros(
            (self._full_size * self._nodes + 4 * (self._nodes - 1),),
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
            pdb.set_trace()

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
            )
        )

        # Declare Update Event: Solve Quadratic Program
        def on_periodic(context, event):
            self.solve_qp(context, event)

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
    """ Seperate All Jit Functions for now """
    # TODO: Params -> PyTreeNode dataclass

    # Cannot figure this out.
    # @staticmethod
    # @jax.jit
    # def _hessian(f):
    #     return jacfwd(jacrev(f))

    # @staticmethod
    # @jax.jit
    # def _compute_hessian(f: Callable, x: jax.Array, y: jax.Array) -> jnp.ndarray:
    #     arr = jacfwd(grad(f))(x, y)
    #     return arr


    @staticmethod
    @partial(jit, static_argnames=['mass', 'friction', 'limit', 'dt'])
    def _equality_constraints(q: jax.Array, initial_conditions: jax.Array, mass: float, friction: float, limit: float, dt: float) -> jnp.ndarray:
        """
        Helper Functions:
            1. Convert Acceleration Data to Control
            2. Hermite-Simpson Collocation
        """
        # Convert Acceleration Data to Control:
        def _compute_control(ddq: float, dq: float, mass: float, friction: float, limit: float) -> float:
            def _convert_to_control(ddq: float, dq: float, mass: float, friction: float) -> float:
                return (ddq * mass + friction * dq)
            u = _convert_to_control(ddq, dq, mass, friction)
            return jax.lax.cond(
                jnp.abs(u) > limit, 
                lambda x: jnp.sign(x) * limit, 
                lambda x: x, 
                u,
            )

        # Hermite-Simpson Collocation  :      
        def _collocation_constraint(x: jax.Array, dt: float) -> jnp.ndarray:
            collocation = x[0][1:] - x[0][:-1] - x[1][:-1] * dt
            midpoint = x[2][:] - (1.0 / 2.0) * (x[0][:-1] + x[0][1:]) + (1 / 8) * dt * (x[1][:-1] - x[1][1:])
            return jnp.concatenate([collocation, midpoint], axis=0)

        """
        Equality Constraints:
            1. Initial Condition
            2. Collocation Constraint
        """
        # TODO: Params passed as PyTree
        _num_states = 6
        _num_nodes = 21

        # Sort State Vector:
        q_m = q[(_num_states * _num_nodes):]
        q = q[:(_num_states * _num_nodes)]

        q = q.reshape((_num_states, -1))
        q_m = q_m.reshape((_num_states - 2, -1))

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

        # Initial Conditions:
        # Conver Acceleration to Control Input:
        ux_ic = _compute_control(
            ddq=initial_conditions[-2],
            dq=initial_conditions[2],
            mass=mass,
            friction=friction,
            limit=limit,
        )
        uy_ic = _compute_control(
            ddq=initial_conditions[-1],
            dq=initial_conditions[3],
            mass=mass,
            friction=friction,
            limit=limit,
        )

        # 1. Initial Condition Constraints:
        initial_condition = jnp.array([
            x[0] - initial_conditions[0],
            y[0] - initial_conditions[1],
            dx[0] - initial_conditions[2],
            dy[0] - initial_conditions[3],
            ux[0] - ux_ic,
            uy[0] - uy_ic,
        ], dtype=float)
        
        # 2. Collocation Constraints:
        ddx = (ux - friction * dx) / mass
        ddy = (uy - friction * dy) / mass
        x_defect = _collocation_constraint([x, dx, x_m], dt)
        y_defect = _collocation_constraint([y, dy, y_m], dt)
        dx_defect = _collocation_constraint([dx, ddx, dx_m], dt)
        dy_defect = _collocation_constraint([dy, ddy, dy_m], dt)

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

    @staticmethod
    @jax.jit
    def _inequality_constraints(q: jax.Array, state_bounds: jax.Array) -> jnp.ndarray:
        """
        Inquality Constraints:
            1. State Bounds
        """
        # TODO: Params passed as PyTree
        _num_states = 6
        _num_nodes = 21

        # Sort State Vector:
        q_m = q[(_num_states * _num_nodes):]
        q = q[:(_num_states * _num_nodes)]

        q = q.reshape((_num_states, -1))
        q_m = q_m.reshape((_num_states - 2, -1))

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

        upper_bounds = jnp.concatenate(
            [
                x - state_bounds[0], x_m - state_bounds[0],
                y - state_bounds[0], y_m - state_bounds[0],
                dx - state_bounds[1], dx_m - state_bounds[1],
                dy - state_bounds[1], dy_m - state_bounds[1],
                ux - state_bounds[2], uy - state_bounds[2],
            ],
            axis=0
        )

        lower_bounds = jnp.concatenate(
            [
                -x - state_bounds[0], -x_m - state_bounds[0],
                -y - state_bounds[0], -y_m - state_bounds[0],
                -dx - state_bounds[1], -dx_m - state_bounds[1],
                -dy - state_bounds[1], -dy_m - state_bounds[1],
                -ux - state_bounds[2], -uy - state_bounds[2],
            ],
            axis=0
        )

        return jnp.vstack([lower_bounds, upper_bounds])

    # Do my mid points need to be included???
    @staticmethod
    @jax.jit
    def _objective_function(q: jax.Array, target_position: jax.Array, w: jax.Array) -> jnp.ndarray:
        """
        Objective Function:
            1. State Error Objective
            2. Control Effort Objective
        """
        # TODO: Params passed as PyTree
        _num_states = 6
        _num_nodes = 21

        # Sort State Vector:
        q = q[:(_num_states * _num_nodes)]
        q = q.reshape((_num_states, -1))

        # State Nodes:
        x = q[0, :]
        y = q[1, :]
        dx = q[2, :]
        dy = q[3, :]
        ux = q[4, :]
        uy = q[5, :]

        error = jnp.vstack(
            [x - target_position[0], y - target_position[1]],
            dtype=float,
        )

        minimize_error = w[0] * jnp.sum(error ** 2, axis=0)
        minimize_effort = w[1] * jnp.sum(ux ** 2 + uy ** 2, axis=0)
        objective_function = jnp.sum(minimize_error + minimize_effort, axis=0)

        return objective_function

    def build_optimization(self, context, event):
        # Get Input Port Values and Convert to Jax Array:
        initial_conditions = jnp.asarray(self.get_input_port(self.initial_condition_input).Eval(context))
        target_positions = jnp.asarray(self.get_input_port(self.target_input).Eval(context))

        # Isolate Functions to Lambda Functions and wrap them in staticmethod:
        self.equality_func = lambda x, ic: self._equality_constraints(
            q=x,
            initial_conditions=ic,
            mass=self._mass,
            friction=self._friction,
            limit=float(self._state_bounds[2]),
            dt=self._dt,
        )

        self.inequality_func = lambda x: self._inequality_constraints(
            q=x,
            state_bounds=self._state_bounds,
        )

        self.objective_func = lambda x, qd: self._objective_function(
            q=x,
            target_position=qd,
            w=self._weights,
        )

        # Compute A and b matricies for equality constraints:
        A_eq = jacfwd(self.equality_func)(self._setpoint, initial_conditions)
        b_eq = self.equality_func(self._setpoint, initial_conditions)

        # Compute A and b matricies for inequality constraints:
        A_ineq = jacfwd(self.inequality_func)(self._setpoint)
        b_ineq = self.inequality_func(self._setpoint)
        
        # Compute H and f matrcies for objective function:
        H = jax.hessian(self.objective_func)(self._setpoint, target_positions)
        
        f = jacfwd(self.objective_func)(self._setpoint, target_positions)
        # f = f.reshape(f.shape[0] * f.shape[1], -1)
        pdb.set_trace()

    # Methods:
    def solve_qp(self, context, event):
        """
        2D Integrator
        """
        # Initialize MathematicalProgram:
        self.prog = MathematicalProgram()

        # Model Parameters:
        # mass = 0.486
        mass = 0.027 # Actual Crazyflie Mass
        friction = 0.01

        # State and Control Input Variables:
        x = self.prog.NewContinuousVariables(self._nodes, "x")
        y = self.prog.NewContinuousVariables(self._nodes, "y")
        dx = self.prog.NewContinuousVariables(self._nodes, "dx")
        dy = self.prog.NewContinuousVariables(self._nodes, "dy")
        ux = self.prog.NewContinuousVariables(self._nodes, "ux")
        uy = self.prog.NewContinuousVariables(self._nodes, "uy")

        # Mid Point Decision Variables for Hermite-Simpson:
        x_m = self.prog.NewContinuousVariables(self._nodes - 1, "x_m")
        y_m = self.prog.NewContinuousVariables(self._nodes - 1, "y_m")
        dx_m = self.prog.NewContinuousVariables(self._nodes - 1, "dx_m")
        dy_m = self.prog.NewContinuousVariables(self._nodes - 1, "dy_m")

        # Create Convenient Arrays:
        _s = np.vstack(np.array([x, y, dx, dy, ux, uy]))

        # Initial Condition Constraints:
        initial_conditions = self.get_input_port(self.initial_condition_input).Eval(context)

        # For Debugging:
        self.initial_conditions = initial_conditions

        # Converts Acceleration to Control and Saturate the Actual Value:
        def compute_control(ddq: float, dq: float) -> float:
            return (ddq * mass + friction * dq)

        def limiter(q: float, saturation_limit: float) -> float:
            if np.abs(q) > saturation_limit:
                q = np.sign(q) * saturation_limit
            return q
        
        limit = 0.05
        initial_conditions[-3] = limiter(compute_control(initial_conditions[-3], initial_conditions[3]), limit)
        initial_conditions[-2] = limiter(compute_control(initial_conditions[-2], initial_conditions[4]), limit)

        """
        If getting full state output of CrazySwarm
        Throw away z indicies
        TO DO: Update to 3D model
        """
        z_index = [2, 5, 8]
        bounds = np.delete(initial_conditions, z_index)
        # [:-4] Unconstrained Acceleration & Velocity
        # Does it make sense to constrain the control input?
        # bounds = bounds[:-2]
        _A_initial_condition = np.eye(np.size(bounds), dtype=float)
        self.prog.AddLinearConstraint(
            A=_A_initial_condition,
            lb=bounds,
            ub=bounds,
            vars=_s[:, 0]
        )

        # Add Lower and Upper Bounds: (Fastest)
        self.prog.AddBoundingBoxConstraint(-5, 5, x)
        self.prog.AddBoundingBoxConstraint(-5, 5, y)
        self.prog.AddBoundingBoxConstraint(-10, 10, dx)
        self.prog.AddBoundingBoxConstraint(-10, 10, dy)
        self.prog.AddBoundingBoxConstraint(-limit, limit, ux)
        self.prog.AddBoundingBoxConstraint(-limit, limit, uy)

        # Collocation Constraints: Python Function
        def hermite_simpson_collocation_func(z):
            _collocation = z[0][1:] - z[0][:-1] - z[1][:-1] * self._dt
            _midpoint = z[2][:] - (1.0 / 2.0) * (z[0][:-1] + z[0][1:]) + (1 / 8) * self._dt * (z[1][:-1] - z[1][1:])
            return (_collocation, _midpoint)

        ddx, ddy = (ux - friction * dx) / mass, (uy - friction * dy) / mass
        _x_collocation, _x_midpoint = hermite_simpson_collocation_func([x,  dx, x_m])
        _dx_collocation, _dx_midpoint = hermite_simpson_collocation_func([dx, ddx, dx_m])
        _y_collocation, _y_midpoint = hermite_simpson_collocation_func([y,  dy, y_m])
        _dy_collocation, _dy_midpoint = hermite_simpson_collocation_func([dy, ddy, dy_m])
        _expr_array = np.asarray(
            [_x_collocation, _x_midpoint,
            _dx_collocation, _dx_midpoint,
            _y_collocation, _y_midpoint,
            _dy_collocation, _dy_midpoint]
        ).flatten()

        bounds = np.zeros(((8 * (self._nodes-1)),), dtype=float)
        self.prog.AddLinearConstraint(
            _expr_array,
            lb=bounds,
            ub=bounds
            )

        # Objective Function Formulation:
        target_positions = self.get_input_port(self.target_input).Eval(context)
        target_positions = np.reshape(target_positions, (2, 1))

        # For Debugging:
        self._target_positions = target_positions

        _error = _s[:2, :] - target_positions
        _weight_distance, _weight_effort = 1.0, 0.0
        _minimize_distance = _weight_distance * np.sum(_error ** 2, axis=0)
        _minimize_effort = _weight_effort * np.sum(ux ** 2 + uy ** 2, axis=0)
        _cost = np.sum(_minimize_distance + _minimize_effort, axis=0)
        objective_function = self.prog.AddQuadraticCost(
            _cost,
            is_convex=True,
        )

        # Good for debugging:
        # assert objective_function.evaluator().is_convex()

        # Solve the program:
        """OSQP:"""
        osqp = OsqpSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(osqp.solver_id(), "max_iter", 10000)
        solver_options.SetOption(osqp.solver_id(), "polish", True)
        solver_options.SetOption(osqp.solver_id(), "warm_start", True)
        solver_options.SetOption(osqp.solver_id(), "verbose", False)
        self.solution = osqp.Solve(
            self.prog,
            self._warm_start,
            solver_options,
        )

        if not self.solution.is_success():
            print(f"Optimization did not solve!")
            print(f"Solver Status: {self.solution.get_solver_details().status_val}")
            print(f"Objective Function Convex: {objective_function.evaluator().is_convex()}")
            pdb.set_trace()
            

        # Store Solution for Output:
        def compute_acceleration(u: np.ndarray, dx: np.ndarray) -> np.ndarray:
            return np.asarray([(u[:] - friction * dx[:]) / mass]).flatten()

        ddx = compute_acceleration(self.solution.GetSolution(ux), self.solution.GetSolution(dx))
        ddy = compute_acceleration(self.solution.GetSolution(uy), self.solution.GetSolution(dy))

        self._full_state_trajectory = np.vstack([
            self.solution.GetSolution(x), self.solution.GetSolution(y),
            self.solution.GetSolution(dx), self.solution.GetSolution(dy),
            ddx, ddy
            ]).flatten()

        self._warm_start = np.concatenate([
            self.solution.GetSolution(x), self.solution.GetSolution(x_m),
            self.solution.GetSolution(y), self.solution.GetSolution(y_m),
            self.solution.GetSolution(dx), self.solution.GetSolution(dx_m),
            self.solution.GetSolution(dy), self.solution.GetSolution(dy_m),
            self.solution.GetSolution(ux), self.solution.GetSolution(uy)
            ], 
            axis=0
        )

        # How can I clean this up?
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_state.set_value(self._full_state_trajectory)


# Test Instantiation:
if __name__ == "__main__":
    unit_test = QuadraticProgram()
