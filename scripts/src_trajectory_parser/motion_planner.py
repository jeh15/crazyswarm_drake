import numpy as np
import ml_collections

from pydrake.all import MathematicalProgram, IpoptSolver, SolverOptions, OsqpSolver
from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

import pdb


class QuadraticProgram(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict):
        LeafSystem.__init__(self)
        # Class Parameters:
        self._UPDATE_RATE = 1.0 / 10.0

        # Mathematical Program Parameters: Takes from Config
        self._nodes = config.nodes
        self._time_horizon = config.time_horizon
        self._state_dimension = config.state_dimension
        self._dt = config.dt

        # State Size for Optimization: (Seems specific to this implementation shouldnt be a config param)
        self._state_size = self._state_dimension * 2
        self._full_size = self._state_dimension * 3

        # Initialize for T = 0:
        self._full_state_trajectory = np.zeros(
            (self._full_size * self._nodes,),
            dtype=float,
            )

        # Initialize Abstract States:
        # Output: (x, y, dx, dy, ux, uy)
        output_size = np.zeros((self._full_size * self._nodes,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        """ Declare Input: """
        # Full State From CrazySwarm:
        # self.initial_condition_input = self.DeclareVectorInputPort("initial_condition", 9).get_index()

        self.initial_condition_input = self.DeclareVectorInputPort(
            "initial_condition",
            self._full_size,
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
        )

        # Declare Initialization Event:
        def on_initialize(context, event):
            self.solve_qp(context, event)

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

    # Methods:
    def solve_qp(self, context, event):
        """
        2D Integrator
        """
        # Initialize MathematicalProgram:
        self.prog = MathematicalProgram()

        # Model Parameters:
        mass = 0.486
        friction = 0.01

        # State and Control Input Variables:
        x = self.prog.NewContinuousVariables(self._nodes, "x")
        y = self.prog.NewContinuousVariables(self._nodes, "y")
        dx = self.prog.NewContinuousVariables(self._nodes, "dx")
        dy = self.prog.NewContinuousVariables(self._nodes, "dy")
        ux = self.prog.NewContinuousVariables(self._nodes, "ux")
        uy = self.prog.NewContinuousVariables(self._nodes, "uy")

        # Create Convenient Arrays:
        _s = np.vstack(np.array([x, y, dx, dy, ux, uy]))

        # Initial Condition Constraints:
        initial_conditions = self.get_input_port(self.initial_condition_input).Eval(context)

        """
        If getting full state output of CrazySwarm
        Throw away z indicies
        TO DO: Update to 3D model
        """
        # z_index = [2, 5, 8]
        # bounds = np.delete(initial_conditions, z_index)
        # [:-2] Unconstrained Acceleration
        bounds = initial_conditions[:-2]
        _A_initial_condition = np.eye(self._state_size, dtype=float)
        self.prog.AddLinearConstraint(
            A=_A_initial_condition,
            lb=bounds,
            ub=bounds,
            vars=_s[:-2, 0]
        )

        # Add Lower and Upper Bounds: (Fastest)
        self.prog.AddBoundingBoxConstraint(-5, 5, x)
        self.prog.AddBoundingBoxConstraint(-5, 5, y)
        self.prog.AddBoundingBoxConstraint(-100, 100, dx)
        self.prog.AddBoundingBoxConstraint(-100, 100, dy)
        self.prog.AddBoundingBoxConstraint(-100, 100, ux)
        self.prog.AddBoundingBoxConstraint(-100, 100, uy)

        # Collocation Constraints: Python Function
        def collocation_func(z):
            _defect = z[0][1:] - z[0][:-1] - z[1][:-1] * self._dt
            return _defect

        ddx, ddy = (ux - friction * dx) / mass, (uy - friction * dy) / mass
        _x_defect = collocation_func([x,  dx])
        _dx_defect = collocation_func([dx, ddx])
        _y_defect = collocation_func([y,  dy])
        _dy_defect = collocation_func([dy, ddy])
        _expr_array = np.asarray([_x_defect, _dx_defect, _y_defect, _dy_defect]).flatten()

        bounds = np.zeros(4 * (self._nodes-1), dtype=float)
        self.prog.AddLinearConstraint(
            _expr_array,
            lb=bounds,
            ub=bounds
            )

        # Objective Function Formulation:
        target_positions = self.get_input_port(self.target_input).Eval(context)
        target_positions = np.reshape(target_positions, (2, 1))
        _error = _s[:2, :] - target_positions
        _weight_distance, _weight_effort = 100.0, 0.0001
        _minimize_distance = _weight_distance * np.sum(_error ** 2, axis=0)
        _minimize_effort = _weight_effort * np.sum(ux ** 2 + uy ** 2, axis=0)
        _cost = np.sum(_minimize_distance + _minimize_effort, axis=0)
        objective_function = self.prog.AddQuadraticCost(
            _cost,
            is_convex=True,
            )

        # Good for debugging:
        # assert objective_function.evaluator().is_convex()

        # Set Initial Guess: (Check how to warm start via OSQP)
        # self.prog.SetInitialGuess(_s.flatten(), self._full_state_trajectory)

        # Solve the program:
        """OSQP:"""
        osqp = OsqpSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(osqp.solver_id(), "max_iter", 1000)
        solver_options.SetOption(osqp.solver_id(), "polish", True)
        solver_options.SetOption(osqp.solver_id(), "verbose", False)
        self.solution = osqp.Solve(
            self.prog,
            self._full_state_trajectory,
            solver_options,
            )

        if not self.solution.is_success():
            print(self.solution.get_solver_details().ConvertStatusToString())
            pdb.set_trace()

        # Store Solution for Output:
        self._full_state_trajectory = np.vstack([
            self.solution.GetSolution(x), self.solution.GetSolution(y),
            self.solution.GetSolution(dx), self.solution.GetSolution(dy),
            self.solution.GetSolution(ux), self.solution.GetSolution(uy)
            ]).flatten()

        self.trajectory = np.vstack([
            self.solution.GetSolution(x),
            self.solution.GetSolution(y),
            np.zeros((self._nodes,), dtype=float)
            ]).flatten()

        # How can I clean this up?
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_state.set_value(self._full_state_trajectory)


# Test Instantiation:
if __name__ == "__main__":
    unit_test = QuadraticProgram()
