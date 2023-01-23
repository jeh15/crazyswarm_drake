import argparse
import numpy as np
import ml_collections

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# Custom LeafSystems:
# import motion_planner_jax as motion_planner
import motion_planner_jax_euler as motion_planner
import reference_trajectory
import trajectory_parser
import crazyswarm_class


# Convenient Data Class:
def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    # Control Rates:
    config.motion_planner_rate = 1.0 / 50.0
    config.reference_trajectory_rate = 1.0 / 50.0
    config.crazyswarm_rate = 1.0 / 100.0
    # Model Parameters:
    config.nodes = 21                   # (Discretized Points)
    config.control_horizon = 11         # (Node to control to) Not used
    config.state_dimension = 2          # (x, y)
    config.time_horizon = 1.0
    config.dt = config.time_horizon / (config.nodes - 1.0)
    return config

def main(argv=None):
    # Create Config Dict:
    params = get_config()

    # Create a block diagram containing our system.
    builder = DiagramBuilder()

    # Reference Motion:
    driver_reference = reference_trajectory.FigureEight(config=params)
    reference = builder.AddSystem(driver_reference)

    # Motion Planner:
    driver_planner = motion_planner.QuadraticProgram(config=params)
    planner = builder.AddSystem(driver_planner)

    # Trajectory Parser:
    driver_parser = trajectory_parser.TrajectoryParser(config=params)
    parser = builder.AddSystem(driver_parser)

    # Crazyswarm Controller:
    driver_crazyswarm = crazyswarm_class.CrazyswarmSystem(config=params)
    crazyswarm = builder.AddSystem(driver_crazyswarm)

    # Connect Systems:
    # Reference Out -> Motion Planner Target Position
    builder.Connect(
        reference.get_output_port(0),
        planner.get_input_port(driver_planner.target_input)
    )

    # Motion Planner Out -> Parser In
    builder.Connect(
        planner.get_output_port(0),
        parser.get_input_port(0),
    )

    # Parser Out -> Crazyswarm In
    builder.Connect(
        parser.get_output_port(0),
        crazyswarm.get_input_port(0),
    )

    # Crazyswarm Out -> Motion Planner Initial Condition
    builder.Connect(
        crazyswarm.get_output_port(0),
        planner.get_input_port(driver_planner.initial_condition_input),
    )

    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()

    # Create the simulator:
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Simulate System:
    FINAL_TIME = 20.0
    dt = 1.0
    next_time_step = dt

    # w/ while-loop:
    while next_time_step <= FINAL_TIME:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
        try:
            simulator.AdvanceTo(next_time_step)
            next_time_step += dt
        except:
            print(f"Exception Occurred...")
            driver_crazyswarm.execute_landing_sequence()
            break

    # w/o while-loop:
    # simulator.AdvanceTo(FINAL_TIME)
    # print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
             

    # Land the Drone:
    driver_crazyswarm.execute_landing_sequence()


if __name__ == "__main__":
    main()