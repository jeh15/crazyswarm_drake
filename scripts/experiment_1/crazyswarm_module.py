import numpy as np
import ml_collections
import rospy

from pydrake.common.value import Value
from pydrake.systems.framework import (
    LeafSystem,
    PublishEvent,
    TriggerType,
    BasicVector_,
)

from pycrazyswarm import *
from crazyswarm.msg import GenericLogData

import pdb


class CrazyswarmSystem(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = config.crazyswarm_rate             # MATCH TRAJECTORY PARSER

        # Class Parameters:
        self._state_dimension = config.state_dimension
        self._full_size = self._state_dimension * 3    # (x, y, dx, dy, ddx, ddy)
        self.target_height = 0.25
        self.noise_threshold = 0.15

        """ Initialize Abstract States: (Output Only) """
        output_size = np.zeros((9,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)

        # Initialize Values:
        self.position = np.zeros((3,), dtype=float)
        self.estimated_states = np.zeros((6,), dtype=float)
        self.previous_position = np.zeros((3,), dtype=float)
        self.previous_states = np.zeros((6,), dtype=float)
        self.current_state = np.zeros((9,))

        # Declare Input: Control Input Package
        self.DeclareVectorInputPort("drone_trajectory_input", self._full_size)

        # Declare Output: VICON Data Package
        self.DeclareVectorOutputPort(
            "drone_full_state_output",
            9,
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        # Declare Initialization Event to Init CrazySwarm:
        def on_initialize(context, event):
            # Initialize Abstract States:
            self.position = self.cf.position()
            full_state_output = np.concatenate(
                [self.position, self.estimated_states],
                axis=0,
            )
            self.current_state = full_state_output
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(full_state_output)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
            )
        )

        # Declare Update Event: Control Crazyflie
        def periodic_event(context, event):
            input_vector = self.get_input_port(0).Eval(context)
            position = np.array(
                [input_vector[0], input_vector[1], self.target_height],
                dtype=float,
            )
            velocity = np.array(
                [input_vector[2], input_vector[3], 0.0],
                dtype=float,
            )
            acceleration = np.array(
                [input_vector[4], input_vector[5], 0.0],
                dtype=float,
            )
            self.cf.cmdFullState(
                pos=position,
                vel=velocity,
                acc=acceleration,
                yaw=0.0,
                omega=np.zeros((3,), dtype=float),
            )

        self.DeclarePeriodicEvent(
            period_sec=self._CONTROL_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_event,
                )
            )

        # Output Update Event:
        def periodic_output_event(context, event):
            # Get Current VICON Positions:
            self.previous_position = self.position
            self.previous_states = self.estimated_states
            self.position = self.cf.position()
            # Filter new data:
            self.filter()
            # Combine with Estimated States:
            full_state_output = np.concatenate(
                [self.position, self.estimated_states],
                axis=0,
            )
            # Logging Visibility:
            self.current_state = full_state_output
            # Update Abstract State:
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(full_state_output)

        self.DeclarePeriodicEvent(
            period_sec=self._CONTROL_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_output_event,
                )
            )
        
        # ROS Subscriber Callback: Estimated Velocity and Acceleration
        def subscriber_callback(data):
            c = 9.80665
            self.estimated_states = np.array(
                [
                    data.values[0], data.values[1], data.values[2],
                    data.values[3] * c, data.values[4] * c, data.values[5] * c,
                ], 
                dtype=float,
            )

    # Output Port Callback:
    def output_callback(self, context, output):
        a_state = context.get_mutable_abstract_state(self.state_index)
        a_value = a_state.get_mutable_value()
        output.SetFromVector(a_value.get_mutable_value())

    # Landing Sequence:
    def execute_landing_sequence(self):
        # Slow Down Drone to 0 Velocity:
        self.ramp_down()
        # Stop Motors:
        self.cf.cmdStop()

    # Filter:
    def filter(self):
        magnitude = np.linalg.norm(self.position[:2] - self.previous_position[:2])
        if magnitude > self.noise_threshold:
            self.position = self.previous_position
            self.estimated_states = self.previous_states

    # Helper Function to Slow Drone Down:
    def ramp_down(self):
        def ramp_down_vector(x: np.ndarray, num_steps: int) -> np.ndarray:
            return np.linspace(x, np.zeros((x.size,)), num_steps)
        steps = 101
        velocity = self.estimated_states[:3]
        acceleration = self.estimated_states[-3:]
        velocity_vector = ramp_down_vector(velocity, steps)
        acceleration_vector = ramp_down_vector(acceleration, steps)
        for i in range(0, steps):
            self.cf.cmdFullState(
                pos=self.position,
                vel=velocity_vector[i],
                acc=acceleration_vector[i],
                yaw=0.0,
                omega=np.zeros((3,), dtype=float),
            )
            self.timeHelper.sleep(self._CONTROL_RATE)

    def initialize_driver(self):
        # ROS Subscriber Callback: Estimated Velocity and Acceleration
        def subscriber_callback(data):
            c = 9.80665
            self.estimated_states = np.array(
                [
                    data.values[0], data.values[1], data.values[2],
                    data.values[3] * c, data.values[4] * c, data.values[5] * c,
                ], 
                dtype=float,
            )

        # Initialize Crazyflies:
        print(f"Initializing Crazyswarm...")
        self.swarm = Crazyswarm()
        self.cf = self.swarm.allcfs.crazyflies[0]
        if self.cf:
            print(f"Crazyflie connected...")
        else:
            print(f"Crazyflie not connected...")

        # Initialize timeHelper and Target Control Rate:
        print(f"Initializing Crazyflie's Time Helper...")
        self.timeHelper = self.swarm.timeHelper
        if self.timeHelper:
            print(f"Time Helper connected...")
        else:
            print(f"Time Helper not connected...")

        # Define Suscriber Callback for State Estimation:
        rospy.Subscriber("/cf2/log1", GenericLogData, subscriber_callback)

        # Save Ground Position:
        self._land_position = self.cf.position()

        # Take Off Script:
        self.cf.takeoff(targetHeight=self.target_height, duration=1.0)
        self.timeHelper.sleep(1.0)

        # Save initial Position:
        self._initial_position = self.cf.position()

        # Initialize Position:
        self.target_position = self._initial_position
        self.position = self._initial_position