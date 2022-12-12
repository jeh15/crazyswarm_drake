import numpy as np
import time

from pydrake.systems.framework import LeafSystem, PublishEvent, TriggerType

import sys
import os
from os.path import dirname, abspath
dir = os.path.dirname(os.getcwd())
p_dir = dirname(dirname(abspath(__file__)))
sys.path.inset(0, p_dir)
from pycrazyswarm import *


class CrazyswarmSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Parameters:
        self._CONTROL_RATE = 1.0 / 100.0  #1 Hz
        self._RUNTIME_RATE = self._CONTROL_RATE / 2.0

        # Declare Input: Control Input Package
        self.DeclareVectorInputPort("reference_trajectory", 3)

        # Declare Output: VICON Data Package
        self.DeclareVectorOutputPort("position", 3, self.output)
        
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

            # Take Off Script:
            target_height = 0.25
            self.cf.takeoff(targetHeight=target_height, duration=1.0)
            self.timeHelper.sleep(1.0)

            # Save initial Position:
            self._initial_position = self.cf.position()

            # Initialize Position:
            self.pos = self._initial_position

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
                self.pos = input_vector + self._initial_position
                self.cf.cmdPosition(self.pos, yaw=0)

                #Check Runtime Allocation:
                _RUNTIME_FLAG = (time.perf_counter() - _start) > self._RUNTIME_RATE

        self.DeclarePeriodicEvent(period_sec=self._CONTROL_RATE,
                    offset_sec=0.0,
                    event=PublishEvent(
                        trigger_type=TriggerType.kPeriodic,
                        callback=on_periodic,
                        )
                    )

    def __del__(self):
        # Land Sequence:
        self.cf.land(0.0, 1.0, groupMask=0)
        # Stop Motors:
        self.cf.stop(groupMask=0)


    # Output Port Callback:
    def output(self, context, pos_data):
        pos_data.SetFromVector(self.pos)
    

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

