import numpy as np
import ml_collections
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

import pdb

import argparse
import cProfile as profile
from contextlib import contextmanager
import pstats

# Custom LeafSystems:
# import motion_planner_jax as motion_planner
import motion_planner_jax_euler as motion_planner
import reference_trajectory
import trajectory_parser
import crazyswarm_class

@contextmanager
def use_cprofile(output_file):
    """
    Use cprofile in specific context.
    """
    pr = profile.Profile()
    pr.enable()
    yield
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats("tottime", "cumtime")
    stats.dump_stats(output_file)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cprofile", type=str)
    args = parser.parse_args()

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
    dt = 1.0 / 100.0
    next_time_step = dt

    with use_cprofile(args.cprofile):
        while next_time_step <= FINAL_TIME:
            print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
            try:
                simulator.AdvanceTo(next_time_step)
                next_time_step += dt
            except:
                print(f"Exception Occurred...")
                driver_crazyswarm.execute_landing_sequence()
                break

    # with use_cprofile(args.cprofile):
    #     simulator.AdvanceTo(FINAL_TIME)
             

    # Land the Drone:
    driver_crazyswarm.execute_landing_sequence()

    # # Report Average Time: Motion Planner
    # avg_time = np.average(driver_planner.drake_time)\
    #         +np.average(driver_planner.drake_time)\
    #         +np.average(driver_planner.osqp_solve_time)\
    #         +np.average(driver_planner.osqp_polish_time)
    # pp_avg = np.average(driver_parser.pp_time)

    # print(f"Average Motion Planner Time: {avg_time} s")
    # print(f"Average PP Time: {pp_avg} s")
    # pdb.set_trace()

    # # Setup Figure: Initialize Figure / Axe Handles
    # fig, ax = plt.subplots()
    # p, = ax.plot([], [], color='red')
    # ax.axis('equal')
    # ax.set_xlim([-3, 3])  # X Lim
    # ax.set_ylim([-3, 3])  # Y Lim
    # ax.set_xlabel('X')  # X Label
    # ax.set_ylabel('Y')  # Y Label
    # ax.set_title('Parser Animation:')
    # video_title = "simulation"

    # # Initialize Patch:
    # c = Circle((0, 0), radius=0.1, color='cornflowerblue')
    # r = Circle((0, 0), radius=0.1, color='red')
    # ax.add_patch(c)
    # ax.add_patch(r)

    # # Setup Animation Writer:
    # dpi = 300
    # FPS = 20
    # simulation_size = len(reference_history)
    # dt = FINAL_TIME / simulation_size
    # sample_rate = int(1 / (dt * FPS))
    # writerObj = FFMpegWriter(fps=FPS)

    # # Plot and Create Animation:
    # with writerObj.saving(fig, video_title+".mp4", dpi):
    #     for i in range(0, simulation_size, sample_rate):
    #         # Plot Reference Trajectory:
    #         r.center = reference_history[i][0], reference_history[i][1]
    #         # Update Patch:
    #         position = parser_history[i]
    #         motion_plan = np.reshape(motion_planner_history[i], (6, -1))
    #         c.center = position[0], position[1]
    #         p.set_data(motion_plan[0, :], motion_plan[1, :])
    #         # Update Drawing:
    #         fig.canvas.draw()  # Update the figure with the new changes
    #         # Grab and Save Frame:
    #         writerObj.grab_frame()

if __name__ == "__main__":
    main()