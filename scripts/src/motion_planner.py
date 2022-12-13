import numpy as np
import time

from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType

import pdb


class QuadraticProgram(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # Class Parameters:
        self._UPDATE_RATE = 1.0 / 10.0 # 10Hz -> 100 ms

        # Declare Input:
        self.initial_condition_input = self.DeclareVectorInputPort("initial_condition", 9).get_index()
        self.target_input = self.DeclareVectorInputPort("target_position", 2).get_index()

        # Declare Output: Trajectory Info
        """Outputs reference trajectory"""
        self.DeclareVectorOutputPort("trajectory", 3, self.output)
        
        # Declare Initialization Event:
        def on_initialize(context, event):
            self.trajectory = np.array([0.0, 0.0, 0.0], dtype=float)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event: Solve Quadratic Program
        def on_periodic(context, event):
            self.update_qp()
            self.solve_qp()
            pdb.set_trace()

        self.DeclarePeriodicEvent(period_sec=self._UPDATE_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=on_periodic,
                        )
                    )

    # Output Port Callback:
    def output(self, context, get_trajectory):
        get_trajectory.SetFromVector(self.trajectory)

    # Methods:
    def update_qp(self, context, event):
        # Initialize MathematicalProgram:
        self.prog = MathematicalProgram()

        # Program Parameters:
        dim_size = 2
        state_size = dim_size * 2
        full_size = dim_size * 3
        num_nodes = 21
        time_horizon = 1.0
        dt = time_horizon / (num_nodes - 1)

        # Model Parameters:
        mass = 0.486
        friction = 0.1 

        # State and Control Inpute Variables:
        x   = self.prog.NewContinuousVariables(num_nodes, "x")
        y   = self.prog.NewContinuousVariables(num_nodes, "y")
        dx  = self.prog.NewContinuousVariables(num_nodes, "dx")
        dy  = self.prog.NewContinuousVariables(num_nodes, "dy")
        ux  = self.prog.NewContinuousVariables(num_nodes, "ux")
        uy  = self.prog.NewContinuousVariables(num_nodes, "uy")

        # Create Convenient Arrays:
        q = np.vstack(np.array([x, y, dx, dy, ux, uy]))

        # Initial Condition Constraints:
        initial_conditions = self.initial_condition_input.Eval(context)
        """
        Throw away z indicies
        TO DO: Update to 3D model
        """
        z_index = [2, 5, 8]
        bounds = np.delete(initial_conditions, z_index)
        _A_initial_condition = np.eye(full_size, dtype=float)
        constraint_initial_condition = self.prog.AddLinearConstraint(
            A=_A_initial_condition,
            lb=bounds,
            ub=bounds,
            vars=q[:, 0]
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
            _defect = z[0][1:] - z[0][:-1] - z[1][:-1] * dt
            return _defect

        ddx, ddy = (ux - friction * dx) / mass, (uy - friction * dy) / mass
        _x_defect   = collocation_func([x,  dx])
        _dx_defect  = collocation_func([dx, ddx])
        _y_defect   = collocation_func([y,  dy])
        _dy_defect  = collocation_func([dy, ddy])
        _expr_array = np.asarray([_x_defect, _dx_defect, _y_defect, _dy_defect]).flatten()

        bounds = np.zeros(4 * (num_nodes-1), dtype=float)
        defect_constraint = self.prog.AddLinearConstraint(
            _expr_array, 
            lb=bounds, 
            ub=bounds 
            )

        # Objective Function:
        target_positions = self.target_input.Eval(context)
        _error = target_positions - q[:2, :]
        _objective_task = self.prog.AddQuadraticCost(np.sum(_error ** 2))

    def solve_qp(self):
        # Solve the program.
        self.solution = Solve(self.prog)
        pdb.set_trace()

    """
    TO DO:
        1. Add trajectory storage method for crazyfly to parse
        2. Add state approximations in crazyfly for full state
        3. Add target reference
        4. Figure out how to regulate the motion planner update
    """
