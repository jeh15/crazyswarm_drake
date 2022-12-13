import numpy as np
import time

from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType
from pycrazyswarm import *

import rospy
from crazyswarm.msg import GenericLogData


class CrazyswarmSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = 1.0 / 100.0  #1 Hz
        self._RUNTIME_RATE = self._CONTROL_RATE / 2.0

        # Declare Input: Control Input Package
        self.DeclareVectorInputPort("reference_trajectory", 3)

        # Declare Output: VICON Data Package
        self.DeclareVectorOutputPort("position", 9, self.output)
        
        # Declare Initialization Event to Init CrazySwarm:
        def on_initialize(context, event):
            # Initialize Crazyflies:
            print(f"Initializing Crazyswarm...")
            self.swarm = Crazyswarm()
            self.cf = self.swarm.allcfs.crazyflies[0]
            if self.cf:
                print(f"Crazyflie connected...")
            else:
                print(f"Crazyflie not connected...")

            # Initialize timeHelper and Target Control Rate:
            print(f"Initializing Time Helper...")
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
            target_height = 0.25
            self.cf.takeoff(targetHeight=target_height, duration=1.0)
            self.timeHelper.sleep(1.0)

            # Save initial Position:
            self._initial_position = self.cf.position()

            # Initialize Position:
            self.target_position = self._initial_position
            self.position = self._initial_position

        self.DeclareInitializationEvent(
            event=PublishEvent(
                trigger_type=TriggerType.kInitialization,
                callback=on_initialize,
                )
            )

        # Declare Update Event: (Control Crazyflie, Get Crazyflie Position)
        def on_periodic(context, event):
            _start = time.perf_counter()
            _RUNTIME_FLAG = False
            while not _RUNTIME_FLAG:
                input_vector = self.get_input_port().Eval(context)
                self.target_position = input_vector + self._initial_position
                self.cf.cmdPosition(self.target_position , yaw=0)

                #Check Runtime Allocation:
                _RUNTIME_FLAG = (time.perf_counter() - _start) > self._RUNTIME_RATE

        self.DeclarePeriodicEvent(period_sec=self._CONTROL_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=on_periodic,
                        )
                    )

        # ROS Subscriber Callback: Estimated Velocity and Acceleration
        def subscriber_callback(data):
            _unit_conversion = 9.80665
            self.estimated_states = np.array(
                [   data.values[0]                   , data.values[1]                   , data.values[2],
                    data.values[3] * _unit_conversion, data.values[4] * _unit_conversion, data.values[5] * _unit_conversion],
                dtype=float,
            )


    # Output Port Callback:
    def output(self, context, output_data):
        # Get Current VICON Positions:
        self.position = self.cf.position()
        # Combine with Estimated States:
        self.full_state = np.concatenate([self.position, self.estimated_states], axis=0, dtype=float)
        # Send Full State Estimates to Output:
        output_data.SetFromVector(self.full_state)
    
    # Landing Sequence:
    def execute_landing_sequence(self):
        # Land Position:
        target_height = self._land_position[-1]
        # Land Sequence:
        self.cf.notifySetpointsStop()
        self.cf.land(targetHeight=target_height, duration=3.0)
        self.timeHelper.sleep(3.0)
        # Stop Motors:
        self.cf.cmdStop()

    """
    TODO: Add this as update event
    """
    # # Command Trajectory:
    # def execute_trajectory(self, trajpath):
    #     _traj = uav_trajectory.Trajectory()
    #     _traj.loadcsv(trajpath)

    #     _start_time = self.timeHelper.time()
    #     while not self.timeHelper.isShutdown():
    #         _loop_start = self.timeHelper.time()
    #         _t = _loop_start - _start_time
    #         if _t > _traj.duration:
    #             break

    #         _e = _traj.eval(_t)
    #         self.cf.cmdFullState(
    #             _e.pos,
    #             _e.vel,
    #             _e.acc,
    #             _e.yaw,
    #             _e.omega)
            
    #         _loop_end = self.timeHelper.time()
    #         _control_regulation =  self.rate - (_loop_end -  _loop_start)
    #         self.timeHelper.sleepForRate(1.0 / _control_regulation)

