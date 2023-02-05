import argparse
import numpy as np
import ml_collections

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import pdb

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# Custom LeafSystems:
import motion_planner_module as motion_planner
import reference_trajectory_module as reference_trajectory
import trajectory_parser_module as trajectory_parser
import crazyswarm_module as crazyswarm_controller
import adversary_tracker_module as adversary_tracker
import risk_learning_module as learning_framework


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

    # Motion Planner:
    driver_planner = motion_planner.QuadraticProgram(config=params)
    planner = builder.AddSystem(driver_planner)

    # Trajectory Parser:
    driver_parser = trajectory_parser.TrajectoryParser(config=params)
    parser = builder.AddSystem(driver_parser)

    # Crazyswarm Controller:
    driver_crazyswarm = crazyswarm_controller.CrazyswarmSystem(config=params)
    crazyswarm = builder.AddSystem(driver_crazyswarm)

    # Adversary Tracker:
    driver_adversary = adversary_tracker.Adversary(config=params)
    adversary = builder.AddSystem(driver_adversary)

    # Risk Learning:
    driver_regression = learning_framework.RiskLearning(config=params)
    regression = builder.AddSystem(driver_regression)

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

    # Adversary Tracker -> Motion Planner Obstacle Initial Condition
    builder.Connect(
        adversary.get_output_port(0),
        planner.get_input_port(driver_planner.obstacle_states_input)
    )

    # Crazyswarm Out -> Learning Framework
    builder.Connect(
        crazyswarm.get_output_port(0),
        regression.get_input_port(driver_regression.agent_input)
    )

    # Adversary Out -> Learning Framework
    builder.Connect(
        adversary.get_output_port(0),
        regression.get_input_port(driver_regression.adversary_input)
    )

    # Learning Framework -> Motion Planner Constraints:
    builder.Connect(
        regression.get_output_port(0),
        planner.get_input_port(driver_planner.constraint_input)
    )

    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()

    # Create the simulator:
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Simulate System:
    FINAL_TIME = 30.0
    dt = 0.1
    next_time_step = dt

    # Lists to store data for animation:
    motion_planner_history = []
    reference_history = []
    parser_history = []
    adversary_history = []
    fp_history = []
    ls_history = []
    data_history = []

    # w/ while-loop:
    while next_time_step <= FINAL_TIME:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
        motion_planner_history.append(driver_planner._full_state_trajectory)
        reference_history.append(driver_reference._reference_trajectory)
        parser_history.append(driver_parser._current_trajectory)
        adversary_history.append(driver_adversary._state_output)
        fp_history.append(driver_regression._fp_sol)
        ls_history.append(driver_regression._ls_sol)
        data_history.append(driver_regression._data)
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

    # Helper function to show reference trajectory:
    def figure_eight_trajectory():
        _r = 1.0
        _time = np.linspace(0, 2 * np.pi)
        _x = _r * np.cos(_time - np.pi / 2.0)
        _y = _r / 2.0 * np.sin(2 * _time)
        return _x, _y

    # Plot results:
    fig, ax = plt.subplots(3)
    fig.tight_layout(pad=2.5)
    ref_plot, = ax[0].plot([], [], color='black', alpha=0.1, linewidth=0.5)
    planner_plot, = ax[0].plot([], [], color='cornflowerblue', alpha=0.5, linewidth=1.0)
    data_plot, = ax[1].plot([], [], color='black', marker='.', linestyle='None')
    fp, = ax[1].plot([], [], color='red')
    ls, = ax[2].plot([], [], color='red')
    # Simulation Plot:
    ax[0].axis('equal')
    ax[0].set_xlim([-3, 3])  # X Lim
    ax[0].set_ylim([-3, 3])  # Y Lim
    ax[0].set_xlabel('X')  # X Label
    ax[0].set_ylabel('Y')  # Y Label
    # FP Plot:
    ax[1].set_xlim([-1, 3])  # X Lim
    ax[1].set_ylim([-1, 2])  # Y Lim
    ax[1].set_xlabel('delta')  # X Label
    ax[1].set_ylabel('r(delta)')  # Y Label
    # LS Plot:
    ax[2].set_xlim([-1, 3])  # X Lim
    ax[2].set_ylim([-2, 1])  # Y Lim
    ax[2].set_xlabel('delta')  # X Label
    ax[2].set_ylabel('s(delta)')  # Y Label

    # Animation
    ax[0].set_title('Risk Learning Animation:')
    video_title = "hardware_risk_learning"

    # Initialize Patches and Plots:
    ref = Circle((0, 0), radius=0.05, color='black')
    adv = Circle((0, 0), radius=0.01, color='red')
    agn = Circle((0, 0), radius=0.01, color='cornflowerblue')
    rad = Circle((0, 0), radius=params.failure_radius, color='red', alpha=0.1)
    ax[0].add_patch(adv)
    ax[0].add_patch(agn)
    ax[0].add_patch(rad)

    x_figure_eight, y_figure_eight = figure_eight_trajectory()
    ref_plot.set_data(x_figure_eight, y_figure_eight)

    # Setup Animation Writer:
    dpi = 300
    FPS = 20
    simulation_size = len(fp_history)
    dt = FINAL_TIME / simulation_size
    sample_rate = int(1 / (dt * FPS))
    if sample_rate == 0:
        sample_rate = 1
    writerObj = FFMpegWriter(fps=FPS)

    # Plot and Create Animation:
    with writerObj.saving(fig, video_title+".mp4", dpi):
        for i in range(0, simulation_size, sample_rate):
            # Plot Reference Trajectory:
            ref.center = reference_history[i][0], reference_history[i][1]
            # Plot Adversary:
            adv.center = adversary_history[i][0], adversary_history[i][1]
            rad.center = adversary_history[i][0], adversary_history[i][1]
            # Plot Agent and Motion Planner Trajectory:
            position = parser_history[i]
            motion_plan = np.reshape(motion_planner_history[i], (6, -1))
            agn.center = position[0], position[1]
            planner_plot.set_data(motion_plan[0, :], motion_plan[1, :])
            # Plot FP:
            data_plot.set_data(data_history[i][0, :], data_history[i][1, :])
            fp.set_data(fp_history[i][0, :], fp_history[i][1, :])
            # Plot LS:
            ls.set_data(ls_history[i][0, :], ls_history[i][1, :])
            # Update Drawing:
            fig.canvas.draw()  # Update the figure with the new changes
            # Grab and Save Frame:
            writerObj.grab_frame()

if __name__ == "__main__":
    main()