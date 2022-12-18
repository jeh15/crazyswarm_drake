import numpy as np

from pydrake.all import MathematicalProgram, Solve
from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

import pdb


class QuadraticProgram(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Class Parameters:
        self._UPDATE_RATE = 1.0 / 5.0

        # Mathematical Program Parameters:
        self._dim_size = 2
        self._state_size = self._dim_size * 2
        self._full_size = self._dim_size * 3
        self._num_nodes = 21
        self._time_horizon = 1.0
        self._dt = self._time_horizon / (self._num_nodes - 1)

        # Initialize Abstract States:
        output_position = 3  # Only outputing the position trajectory (x, y, z)
        output_size = np.zeros((output_position * self._num_nodes,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        # Declare Input:
        # Full State From CrazySwarm:
        # self.initial_condition_input = self.DeclareVectorInputPort("initial_condition", 9).get_index()
        # Test Input:
        self.initial_condition_input = self.DeclareVectorInputPort("initial_condition", 6).get_index()
        self.target_input = self.DeclareVectorInputPort("target_position", 2).get_index()

        # Declare Output: Trajectory Info
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
        friction = 0.1

        # State and Control Input Variables:
        x = self.prog.NewContinuousVariables(self._num_nodes, "x")
        y = self.prog.NewContinuousVariables(self._num_nodes, "y")
        dx = self.prog.NewContinuousVariables(self._num_nodes, "dx")
        dy = self.prog.NewContinuousVariables(self._num_nodes, "dy")
        ux = self.prog.NewContinuousVariables(self._num_nodes, "ux")
        uy = self.prog.NewContinuousVariables(self._num_nodes, "uy")

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
        bounds = initial_conditions
        _A_initial_condition = np.eye(self._full_size, dtype=float)
        self.prog.AddLinearConstraint(
            A=_A_initial_condition,
            lb=bounds,
            ub=bounds,
            vars=_s[:, 0]
        )

        # Add Lower and Upper Bounds: (Fastest)
        self.prog.AddBoundingBoxConstraint(-5, 5, x)
        self.prog.AddBoundingBoxConstraint(-5, 5, y)
        self.prog.AddBoundingBoxConstraint(-5, 5, dx)
        self.prog.AddBoundingBoxConstraint(-5, 5, dy)
        self.prog.AddBoundingBoxConstraint(-1, 1, ux)
        self.prog.AddBoundingBoxConstraint(-1, 1, uy)

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

        bounds = np.zeros(4 * (self._num_nodes-1), dtype=float)
        self.prog.AddLinearConstraint(
            _expr_array,
            lb=bounds,
            ub=bounds
            )

        # Objective Function:
        target_positions = self.get_input_port(self.target_input).Eval(context)
        target_positions = np.reshape(target_positions, (2, 1))
        _error = target_positions - _s[:2, :]
        self.prog.AddQuadraticCost(np.sum(_error ** 2))

        # Solve the program.
        self.solution = Solve(self.prog)

        # Store Solution for Output:
        self._full_state_trajectory = np.vstack([
            self.solution.GetSolution(x), self.solution.GetSolution(y),
            self.solution.GetSolution(dx), self.solution.GetSolution(dy),
            self.solution.GetSolution(ux), self.solution.GetSolution(uy)
            ]).flatten()

        self.trajectory = np.vstack([
            self.solution.GetSolution(x),
            self.solution.GetSolution(y),
            np.zeros((self._num_nodes,), dtype=float)
            ]).flatten()

        # How can I clean this up?
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_state.set_value(self.trajectory)


# Test Instantiation:
if __name__ == "__main__":
    unit_test = QuadraticProgram()
