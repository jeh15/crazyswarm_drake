import numpy as np
import timeit
import ml_collections

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

from pycrazyswarm import *
import motioncapture

import pdb


class Adversary(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = config.crazyswarm_rate  
        self._OUTPUT_UPDATE_RATE = config.motion_planner_rate   # MATCH MOTION PLANNER

        # Class Parameters:
        self._state_dimension = config.state_dimension
        self._full_size = self._state_dimension * 3    # (x, y, dx, dy, ddx, ddy)

        # Initialize on Class Instantiation:
        self.mc = motioncapture.connect("vicon", "192.168.1.119")
        self.swarm = Crazyswarm()
        self.timeHelper = self.swarm.timeHelper

        # Initialize Values:
        self.position = np.zeros((3,), dtype=float)
        self.estimated_states = np.zeros((3,), dtype=float)
        self._state_output = np.concatenate([self.position, self.estimated_states])

        """ Initialize Abstract States: (Output Only) """
        output_size = np.ones((self._full_size,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        # Declare Output: VICON Data Package
        self.DeclareVectorOutputPort(
            "adversary_state_output",
            output_size.shape[0],
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        # Declare Initialization Event to Init CrazySwarm:
        def on_initialize(context, event):
            # Get Initial Velocity:
            self.finite_difference()
            state_output = np.concatenate([self.position, self.estimated_states], axis=0)

            # Initialize Abstract States:
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(state_output)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

            # Output Update Event:
        def periodic_output_event(context, event):
            # Get Position and Estimated States:
            self.finite_difference()
            state_output = np.concatenate([self.position, self.estimated_states], axis=0)

            self._state_output = state_output

            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(state_output)

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
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_value = a_state.get_mutable_value()
        output.SetFromVector(a_value.get_mutable_value())

    # Finite Differencing for Adversary States:
    def finite_difference(self) -> None:
        self.mc.waitForNextFrame()
        adversary = self.mc.rigidBodies['stick']
        wall_t_start = timeit.default_timer()
        initial_position = adversary.position.copy()
        self.mc.waitForNextFrame()
        adversary = self.mc.rigidBodies['stick']
        self.position = adversary.position.copy()
        dt = timeit.default_timer() - wall_t_start
        self.estimated_states = (self.position - initial_position) / dt