import numpy as np
import ml_collections

# DEBUG:
import pdb
import timeit

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)
from pydrake.trajectories import PiecewisePolynomial


class TrajectoryParser(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Class Parameters: (Could be contained in Config)
        self._INPUT_UPDATE_RATE = config.motion_planner_rate    # MATCH MOTION PLANNER
        self._OUTPUT_UPDATE_RATE = config.crazyswarm_rate       # MATCH CRAZYSWARM

        # Make Config Params Class Variables:
        self._nodes = config.nodes
        self._time_horizon = config.time_horizon
        self._state_dimension = config.state_dimension
        self._dt = config.dt

        # Class Specific Parameters:
        self._full_size = self._state_dimension * 3    # (x, y, dx, dy, ddx, ddy)
        self._time_trajectory = np.linspace(0.0, self._time_horizon, self._nodes)

        """ Initialize Abstract States: (Output Only) """
        # Motion Planner is 2D
        output_size = np.zeros((self._full_size,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        """Input Motion Planner trajectory"""
        # Motion Planner is 2D
        self.DeclareVectorInputPort("parser_input", self._full_size * self._nodes)

        # Declare Output: Trajectory Info
        """Outputs reference trajectory"""
        self.DeclareVectorOutputPort(
            "parser_output",
            self._full_size,
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        """ Declare Initialization Event: Default Values """
        def on_initialize(context, event):
            self._trajectory_time = 0.0
            # Motion Planner is 2D:
            self.trajectory = PiecewisePolynomial.FirstOrderHold(
                breaks=self._time_trajectory[1:],
                samples=np.zeros((self._full_size, self._nodes - 1), dtype=float),
            )

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        """ Declare Update Event: """
        # (Skip First Node)
        def periodic_input_event(context, event):
            # Get Current Time:
            self._trajectory_time = context.get_time()
            # Read Input Trajectory for Storage:
            motion_plan = self.get_input_port(0).Eval(context)
            # Reshape the motion plan for easy manipulation:
            motion_plan = np.reshape(motion_plan, (-1, self._nodes))
            dq = motion_plan[2:4, :]
            ddq = motion_plan[4:, :]
            dddq = np.gradient(ddq, self._time_trajectory, axis=1)
            motion_plan_dot = np.concatenate([dq, ddq, dddq], axis=0)
            # Construct motion plan into Polynomial for interpolation:
            self.trajectory = PiecewisePolynomial.CubicHermite(
                breaks=self._time_trajectory[1:],
                samples=motion_plan[:, 1:],
                samples_dot=motion_plan_dot[:, 1:],
            )

        self.DeclarePeriodicEvent(
            period_sec=self._INPUT_UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_input_event,
            ),
        )

        # Declare Update Event:
        def periodic_output_event(context, event):
            # Interpolate trajectory to get current value:
            current_time = context.get_time() - self._trajectory_time

            # Interpolate from Polynomial:
            current_trajectory = self.trajectory.value(current_time)

            # For Debugging:
            self._current_trajectory = current_trajectory
            
            # Update Abstract State:
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(current_trajectory)

        self.DeclarePeriodicEvent(
            period_sec=self._OUTPUT_UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_output_event,
            ),
        )

    # Output Port Callback:
    def output_callback(self, context, output):
        # output.SetFromVector(self.current_trajectory)
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_value = a_state.get_mutable_value()
        self._output = a_value
        output.SetFromVector(a_value.get_mutable_value())
