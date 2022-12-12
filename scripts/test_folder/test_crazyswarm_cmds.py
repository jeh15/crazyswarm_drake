import numpy as np

from pydrake.systems.framework import LeafSystem
from pycrazyswarm import *
import uav_trajectory


class CrazyswarmSystem(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        # Declare Input: Control Input Package
        """Gets full state trajectory from DirectCollocation"""

        # Declare Output: VICON Data Package
        """Outputs trajectory of drone during execute_trajectory"""
        
        # Declare Update Event: (Control Crazyflie, Get Crazyflie Position)
        """
        Declare events(?methods?) that must be called while diagram is running
        Example: get_position and exectute_trajectory
        """

        # Initialize Crazyflies:
        self.swarm = Crazyswarm()
        self.cf = self.swarm.allcfs.crazyflies[0]

        # Initialize timeHelper and Target Control Rate:
        self.timeHelper = self.swarm.timeHelper
        self.rate = 1.0 / 100.0 # 10ms

    def get_position(self):
        # ADD Vicon Commands Here
        self.pos = [0.0, 0.0, 0.0]

    # Command Trajectory:
    def execute_trajectory(self, trajpath):
        _traj = uav_trajectory.Trajectory()
        _traj.loadcsv(trajpath)

        _start_time = self.timeHelper.time()
        while not self.timeHelper.isShutdown():
            _loop_start = self.timeHelper.time()
            _t = _loop_start - _start_time
            if _t > _traj.duration:
                break

            _e = _traj.eval(_t)
            self.cf.cmdFullState(
                _e.pos,
                _e.vel,
                _e.acc,
                _e.yaw,
                _e.omega)
            
            _loop_end = self.timeHelper.time()
            _control_regulation =  self.rate - (_loop_end -  _loop_start)
            self.timeHelper.sleepForRate(1.0 / _control_regulation)


# Main Function:
def main():
    DroneSystem = CrazyswarmSystem()

# Test:
if __name__ == "__main__":
    main()