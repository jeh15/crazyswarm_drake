import numpy as np
import time

# DEBUG:
import pdb

from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType
from pydrake.trajectories import PiecewisePolynomial


class TrajectoryParser(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Class Parameters:
        self._INPUT_UPDATE_RATE = 1.0 / 5.0     # MATCH MOTION PLANNER
        self._OUTPUT_UPDATE_RATE = 1.0 / 100.0  # MATCH CRAZYSWARM

        # **THESE VALUES CHANGE RELATIVE TO THE MOTION PLANNER**
        self._num_dims = 2
        self._num_states = 2 * 3
        self._num_nodes = 21
        self._time_horizon = 1.0
        self._time_trajectory = np.linspace(0.0, self._time_horizon, self._num_nodes)

        # Declare Input:
        """Input Motion Planner trajectory"""
        self.DeclareVectorInputPort("parser_input", 63)

        # Declare Output: Trajectory Info
        """Outputs reference trajectory"""
        self.DeclareVectorOutputPort(
            "parser_output", 
            3, 
            self.output_callback,
            {self.nothing_ticket()})
        
        # Declare Initialization Event: Default Values
        def on_initialize(context, event):
            self.current_trajectory = np.array([0.0, 0.0, 0.0], dtype=float)
            self._trajectory_time = 0.0
            self.trajectory = PiecewisePolynomial.FirstOrderHold(
                breaks=self._time_trajectory,
                samples=np.zeros((self._num_dims, self._num_nodes), dtype=float)
                )

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event:
        def periodic_input_event(context, event):
            # Get Current Time:
            self._trajectory_time = context.get_time()
            # Read Input Trajectory for Storage:
            _motion_plan = self.get_input_port(0).Eval(context)
            # Reshape the motion plan for easy manipulation:
            _motion_plan = np.reshape(_motion_plan, (-1, self._num_nodes))
            # Construct motion plan into Polynomial for interpolation:
            self.trajectory = PiecewisePolynomial.FirstOrderHold(
                breaks=self._time_trajectory,
                samples=_motion_plan[:2, :]
                )

        self.DeclarePeriodicEvent(period_sec=self._INPUT_UPDATE_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=periodic_input_event,
                        )
                    )

        # Declare Update Event:
        def periodic_output_event(context, event):
            # Interpolate trajectory to get current value:
            _current_time = context.get_time() - self._trajectory_time
            # Interpolate from Polynomial:
            _current_trajectory = self.trajectory.value(_current_time)
            _current_trajectory = np.vstack([_current_trajectory, [0.0]]).flatten()
            self.current_trajectory = _current_trajectory

        self.DeclarePeriodicEvent(period_sec=self._OUTPUT_UPDATE_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=periodic_output_event,
                        )
                    )

    # Output Port Callback:
    def output_callback(self, context, output):
        output.SetFromVector(self.current_trajectory)

    # Methods:
    def _figure_eight_trajectory(self, context, event):
        _r = 1.0
        _time = context.get_time()
        _x = _r * np.cos(_time - np.pi / 2.0)
        _y = _r / 2.0 * np.sin(2 * _time)
        return _x, _y