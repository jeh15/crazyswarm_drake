import numpy as np
import ml_collections

# DEBUG:
import pdb

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
        self._samples = config.samples

        # Class Specific Parameters:
        self._full_size = self._state_dimension * 3    # (x, y, dx, dy, ddx, ddy)
        self._time_trajectory = np.linspace(0.0, self._time_horizon, self._nodes)
        self.previous_trajectory = np.zeros((self._full_size, self._samples.shape[0]), dtype=float)

        """ Initialize Abstract States: (Output Only) """
        control_output_size = np.zeros((self._full_size,))
        control_output_state_init = Value[BasicVector_[float]](control_output_size)
        self.control_state_index = self.DeclareAbstractState(control_output_state_init)

        previous_trajectory_output_size = np.zeros((self._full_size * self._samples.shape[0],))
        previous_trajectory_output_state_init = Value[BasicVector_[float]](previous_trajectory_output_size)
        self.previous_trajectory_state_index = self.DeclareAbstractState(previous_trajectory_output_state_init)

        """Input Motion Planner trajectory"""
        self.DeclareVectorInputPort("parser_input", self._full_size * self._nodes)

        # Declare Output: Trajectory Info
        """Outputs reference trajectory"""
        self.control_output = self.DeclareVectorOutputPort(
            "parser_control_output",
            self._full_size,
            self.control_output_callback,
            {self.abstract_state_ticket(self.control_state_index)},
        ).get_index()

        self.previous_trajectory_output = self.DeclareVectorOutputPort(
            "parser_previous_trajectory_output",
            previous_trajectory_output_size.shape[0],
            self.previous_trajectory_output_callback,
            {self.abstract_state_ticket(self.previous_trajectory_state_index)},
        ).get_index()

        """ Declare Initialization Event: Default Values """
        def on_initialize(context, event):
            self._trajectory_time = 0.0
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
            # Store interpolated trajectory at sample times to make continuity soft constraint in optimization:
            previous_trajectory = []
            for i in self._samples:
                previous_trajectory.append(self.trajectory.value(i))
            self.previous_trajectory = np.concatenate(previous_trajectory, axis=1)
            a_state = context.get_mutable_abstract_state(self.previous_trajectory_state_index)
            a_state.set_value(self.previous_trajectory.flatten())

        self.DeclarePeriodicEvent(
            period_sec=self._INPUT_UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_input_event,
            ),
        )

        # Declare Update Event:
        def periodic_control_output_event(context, event):
            # Interpolate trajectory to get current value:
            current_time = context.get_time() - self._trajectory_time

            # Interpolate from Polynomial:
            current_trajectory = self.trajectory.value(current_time)

            # Visiblity for Debugging/Logging:
            self._current_trajectory = current_trajectory

            # Update Abstract State:
            a_state = context.get_mutable_abstract_state(self.control_state_index)
            a_state.set_value(current_trajectory)

        self.DeclarePeriodicEvent(
            period_sec=self._OUTPUT_UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_control_output_event,
            ),
        )

    # Output Port Callback:
    def control_output_callback(self, context, output):
        a_state = context.get_mutable_abstract_state(self.control_state_index)
        a_value = a_state.get_mutable_value()
        output.SetFromVector(a_value.get_mutable_value())

    # Output Port Callback:
    def previous_trajectory_output_callback(self, context, output):
        a_state = context.get_mutable_abstract_state(self.previous_trajectory_state_index)
        a_value = a_state.get_mutable_value()
        output.SetFromVector(a_value.get_mutable_value())
