import numpy as np
import ml_collections

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)


class ReferenceTrajectory(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)

        # Class Parameters:
        self._UPDATE_RATE = config.crazyswarm_rate
        self._full_size = config.state_dimension * 3 

        # Initialize Abstract States: 
        f_state_size = np.zeros((self._full_size,))
        f_state_init = Value[BasicVector_[float]](f_state_size)
        self.f_state_index = self.DeclareAbstractState(f_state_init)
        self._figure_eight_reference = np.zeros((self._full_size,), dtype=float)

        l_state_size = np.zeros((9,))
        l_state_init = Value[BasicVector_[float]](l_state_size)
        self.l_state_index = self.DeclareAbstractState(l_state_init)
        self._linear_reference = np.zeros((self._full_size + 3,), dtype=float)

        # Constant Parameters:
        self.r = 1.0
        self.factor = 4.0

        # Assign Initial Values: (Figure Eight)
        self.previous_x = self.r * np.cos(0.0 - np.pi / 2.0)
        self.previous_y = self.r / 2.0 * np.sin(2 * 0.0)
        self.previous_time = 0.0

        # Assign Initial Values: (Linear Motion)
        self.linear_previous_x = -self.r * np.sin(0)
        self.linear_previous_y = 0.0
        self.linear_previous_time = 0.0

        # Declare Output: Vector
        self.figure_eight_output = self.DeclareVectorOutputPort(
            "figure_eight_target_position",
            np.size(f_state_size),
            self.figure_eight_output_callback,
            {self.abstract_state_ticket(self.f_state_index)},
        ).get_index()

        self.linear_output = self.DeclareVectorOutputPort(
            "linear_target_position",
            np.size(l_state_size),
            self.linear_output_callback,
            {self.abstract_state_ticket(self.l_state_index)},
        ).get_index()

        # Declare Initialization Event:
        def on_initialize(context, event):
            figure_eight_states = self._figure_eight_trajectory(context)
            linear_states = self._linear_trajectory(context)
            # Visibility for Debug/Logging:
            self._figure_eight_reference = figure_eight_states
            self._linear_reference = linear_states
            f_state = context.get_mutable_abstract_state(self.f_state_index)
            f_state.set_value(figure_eight_states)
            l_state = context.get_mutable_abstract_state(self.l_state_index)
            l_state.set_value(linear_states)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event: Current Trajectory
        def on_periodic(context, event):
            figure_eight_states = self._figure_eight_trajectory(context)
            linear_states = self._linear_trajectory(context)
            # Visibility for Debug/Logging:
            self._figure_eight_reference = figure_eight_states
            self._linear_reference = linear_states
            f_state = context.get_mutable_abstract_state(self.f_state_index)
            f_state.set_value(figure_eight_states)
            l_state = context.get_mutable_abstract_state(self.l_state_index)
            l_state.set_value(linear_states)

        self.DeclarePeriodicEvent(
            period_sec=self._UPDATE_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=on_periodic,
                )
            )

    # Output Port Callback:
    def figure_eight_output_callback(self, context, figure_eight_target_position):
        f_state = context.get_mutable_abstract_state(self.f_state_index)
        f_value = f_state.get_mutable_value()
        figure_eight_target_position.SetFromVector(f_value.get_mutable_value())

    def linear_output_callback(self, context, linear_target_position):
        l_state = context.get_mutable_abstract_state(self.l_state_index)
        l_value = l_state.get_mutable_value()
        linear_target_position.SetFromVector(l_value.get_mutable_value())

    # Class Methods:
    def _figure_eight_trajectory(self, context):
        time = context.get_time()
        rate = time / self.factor
        x = self.r * np.cos(rate - np.pi / 2.0)
        y = self.r / 2.0 * np.sin(2 * rate)
        delta_x = x - self.previous_x
        delta_y = y - self.previous_y
        delta_t = time - self.previous_time
        dx = np.divide(delta_x, delta_t, out=np.zeros_like(delta_x, dtype=float), where=delta_t!=0)
        dy = np.divide(delta_y, delta_t, out=np.zeros_like(delta_y, dtype=float), where=delta_t!=0)
        # Assign previous states:
        self.previous_x = x
        self.previous_y = y
        self.previous_time = time 
        return np.array([x, y, 0, 0, 0, 0], dtype=float)

    def _linear_trajectory(self, context):
        time = context.get_time()
        rate = time / self.factor
        x = -self.r * np.sin(rate)
        y = 0.0
        delta_x = x - self.linear_previous_x
        delta_y = y - self.linear_previous_y
        delta_t = time - self.linear_previous_time
        dx = np.divide(delta_x, delta_t, out=np.zeros_like(delta_x, dtype=float), where=delta_t!=0)
        dy = np.divide(delta_y, delta_t, out=np.zeros_like(delta_y, dtype=float), where=delta_t!=0)
        # Assign previous states:
        self.linear_previous_x = x
        self.linear_previous_y = y
        self.linear_previous_time = time
        return np.array([x, y, 0, dx, dy, 0, 0, 0, 0], dtype=float)
