import numpy as np
import timeit
import ml_collections
import copy
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


class Adversary(LeafSystem):
    def __init__(self, config: ml_collections.ConfigDict()):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = config.adversary_rate

        # Class Parameters:
        self._state_dimension = config.state_dimension
        self._full_size = self._state_dimension * 3    # (x, y, dx, dy, ddx, ddy)
        self.previous_position = np.zeros((3,), dtype=float)
        self.noise_threshold = 0.15

        # Debug Variables:
        self._target_position = np.zeros((3,), dtype=float)

        # PID Controller:
        self.saturation_limit = 0.0
        self.saturation_max_limit = 200.0
        self.ramp_time = 10.0
        self._safety_offset = 0.1
        self._error_previous = 0.0
        self._error = np.zeros((3,), dtype=float)
        self._error_derivative = np.zeros((3,), dtype=float)
        
        self._P_limt = 100.0
        self._P_init = 10.0
        self._P = 10.0
        self._D = 20.0

        # Initialize Values:
        self.target_height = 0.25
        self.velocity = np.zeros((3,), dtype=float)
        self._state_output = np.zeros((self._full_size,), dtype=float)

        """ Initialize Abstract States: (Output Only) """
        output_size = np.ones((self._full_size,))
        output_state_init = Value[BasicVector_[float]](output_size)
        self.state_index = self.DeclareAbstractState(output_state_init)
        self._initial_state_output = np.zeros((self._full_size,), dtype=float)

        # # Declare Input: Target States
        self.DeclareVectorInputPort("target_states", 9)

        # Declare Output: VICON Data Package
        self.DeclareVectorOutputPort(
            "adversary_state_output",
            output_size.shape[0],
            self.output_callback,
            {self.abstract_state_ticket(self.state_index)},
        )

        # Declare Initialization Event to Init CrazySwarm:
        def on_initialize(context, event):
            # Initialize Abstract States:
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(self._initial_state_output)

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
            )
        )

        # Periodic Update Event:
        def periodic_update_event(context, event):
            self.pd_control(context)

        self.DeclarePeriodicEvent(
            period_sec=self._CONTROL_RATE,
            offset_sec=0.0,
            event=PublishEvent(
                trigger_type=TriggerType.kPeriodic,
                callback=periodic_update_event,
            )
        )

        # Output Update Event:
        def periodic_output_event(context, event):
            # Get Position and Estimated States:
            position = copy.deepcopy(self.cf.position())
            velocity = copy.deepcopy(self.velocity)
            position, velocity = self.filter(position, velocity)
            self.previous_position = position
            state_output = np.concatenate([position, velocity], axis=0)
            self._state_output = state_output
            a_state = context.get_mutable_abstract_state(self.state_index)
            a_state.set_value(state_output)

        self.DeclarePeriodicEvent(
            period_sec=self._CONTROL_RATE,
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

    def pd_control(self, context) -> None:
        # Get control saturation limit:
        self.limiter(context)

        # Get target position from input:
        target_position = np.asarray((self.get_input_port(0).Eval(context))[:3])

        # Update estimated states:
        position = copy.deepcopy(self.cf.position())
        position, _ = self.filter(position, 0)
        self.previous_position = position

        # Eliminate Z-axis:
        position[-1] = self.target_height
        target_position[-1] = self.target_height

        # For debugging:
        self._target_position = target_position

        # Calculate error vector:
        position_vector = target_position - position
        magnitude = np.linalg.norm(position_vector)
        unit_vector = position_vector / magnitude

        # Calculate new errors:
        error_magnitude = magnitude - self._safety_offset
        self._error = error_magnitude * unit_vector
        self._error_derivative = (error_magnitude - self._error_previous) * unit_vector
        self._error_previous = error_magnitude

        # Calculate control input:
        control_input = (self._P * self._error) + (self._D * self._error_derivative)

        # Saturate Control Inputs:
        for i in range(control_input.shape[0]):
            if np.abs(control_input[i]) > self.saturation_limit:
                control_input[i] = np.sign(control_input[i]) * self.saturation_limit

        # Control along the Z-axis should always be 0:
        control_input[-1] = 0.0

        position_prediction = position + control_input * self._CONTROL_RATE

        self.cf.cmdPosition(
            pos=position_prediction,
            yaw=0.0,
        )
    
    def limiter(self, context):
        context_time = context.get_time()
        if context_time <= self.ramp_time:
            control_input_ramp = [0.0, self.saturation_max_limit]
            gain_schedule = [self._P_init, self._P_limt]
            ramp_time = [0.0, self.ramp_time]
            self.saturation_limit = np.interp(context_time, ramp_time, control_input_ramp)
            self._P = np.interp(context_time, ramp_time, gain_schedule)
        else:
            self._P = self._P_limt
            self.saturation_limit = self.saturation_max_limit

    # Hacky way to filter out random VICON blips:
    def filter(self, position, velocity):
        magnitude = np.linalg.norm(position[:2] - self.previous_position[:2])
        if magnitude > self.noise_threshold:
            position = self.previous_position
            velocity = self.previous_velocity
        return position, velocity

    # Helper Function to Slow Drone Down:
    def ramp_down(self):
        def ramp_down_vector(x: np.ndarray, num_steps: int) -> np.ndarray:
            return np.linspace(x, np.zeros((x.size,)), num_steps)
        position = copy.deepcopy(self.cf.position())
        steps = 101
        velocity = self.velocity
        velocity_vector = ramp_down_vector(velocity, steps)
        position_vector = position + velocity_vector * self._CONTROL_RATE
        for position in position_vector:
            self.cf.cmdPosition(
                pos=position,
                yaw=0.0,
            )
            self.timeHelper.sleep(self._CONTROL_RATE)

    # Landing Sequence:
    def execute_landing_sequence(self):
        # Slow Down Drone to 0 Velocity:
        self.ramp_down()
        # Stop Motors:
        self.cf.cmdStop()
    
    def initialize_driver(self):
        # ROS Subscriber Callback: Estimated Velocity and Acceleration
        def subscriber_callback(data):
            self.velocity = np.array(
                [
                    data.values[0], data.values[1], data.values[2],
                ], 
                dtype=float,
            )

        # Initialize Crazyflies:
        print(f"Initializing Crazyswarm...")
        self.swarm = Crazyswarm()
        self.cf = self.swarm.allcfs.crazyflies[-1]
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
        rospy.Subscriber("/cf4/log1", GenericLogData, subscriber_callback)

        # Save Ground Position:
        self._land_position = self.cf.position()

        # Take Off Script:
        self.cf.takeoff(targetHeight=self.target_height, duration=1.0)
        self.timeHelper.sleep(1.0)

        # Save initial Position:
        self._initial_position = self.cf.position()

        # Compute Finite Difference:
        position = copy.deepcopy(self.cf.position())
        self.previous_position = position
        velocity = copy.deepcopy(self.velocity)
        self.previous_velocity = velocity
        self._initial_state_output = np.concatenate([position, velocity], axis=0)
