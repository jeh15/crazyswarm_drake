import argparse
import time
import numpy as np
import ml_collections

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import pdb

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

# Custom LeafSystems:
import adversary_module as adversary_controller
import reference_trajectory_module as reference_trajectory

# Saving Script:
import save_data


# Convenient Data Class:
def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    # Control Rates:
    config.motion_planner_rate = 1.0 / 40.0
    config.reference_trajectory_rate = 1.0 / 40.0
    config.crazyswarm_rate = 1.0 / 100.0
    config.adversary_rate = 1.0 / 100.0
    # Model Parameters:
    config.nodes = 21                   # (Discretized Points)
    config.control_horizon = 11         # (Node to control to) Not used
    config.state_dimension = 2          # (x, y)
    config.time_horizon = 2.0
    config.dt = config.time_horizon / (config.nodes - 1.0)
    # Spline Parameters:
    config.spline_resolution = 7
    config.bin_resolution = 100
    config.failure_radius = 0.25
    return config

def main(argv=None):
    # Create Config Dict:
    params = get_config()

    # Create a block diagram containing our system.
    builder = DiagramBuilder()

    # Reference Motion:
    driver_reference = reference_trajectory.FigureEight(config=params)
    reference = builder.AddSystem(driver_reference)

    # Adversary Tracker:
    driver_adversary = adversary_controller.Adversary(config=params)
    adversary = builder.AddSystem(driver_adversary)

    # Connect Systems:
    # Target Position -> Adversary
    builder.Connect(
        reference.get_output_port(0),
        adversary.get_input_port(0)
    )

    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()

    # Create the simulator:
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)

    # Initialize Crazyswarm:
    driver_adversary.initialize_driver()

    # Initialize Simulator:
    simulator.Initialize()

    # Simulate System:
    FINAL_TIME = 30.0
    dt = 0.1
    next_time_step = dt

    # w/ while-loop:
    while next_time_step <= FINAL_TIME:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
        try:
            simulator.AdvanceTo(next_time_step)
            next_time_step += dt
        except:
            print(f"Exception Occurred...")
            break

if __name__ == "__main__":
    main()