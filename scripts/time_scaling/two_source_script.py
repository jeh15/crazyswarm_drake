import argparse
import time
import numpy as np
import ml_collections

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import pdb

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# Custom LeafSystems:
import reference_trajectory_module as reference_trajectory
import motion_planner_module as motion_planner
import trajectory_parser_module as trajectory_parser
import crazyswarm_module as crazyswarm_controller
import adversary_module as adversary_controller
import risk_learning_module as learning_framework
import evaluator_extension

# Saving Script:
import shelve_list


# Convenient Data Class:
def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    # Control Rates:
    config.motion_planner_rate = 1.0 / 40.0
    config.crazyswarm_rate = 1.0 / 100.0
    config.adversary_rate = 1.0 / 100.0
    # Model Parameters:
    config.nodes = 21
    config.state_dimension = 2
    config.time_horizon = 1.0
    config.time_vector = np.power(np.linspace(0, config.time_horizon, config.nodes), np.e)
    config.dt_vector = config.time_vector[1:] - config.time_vector[:-1]
    config.area_bounds = 1.5
    # Evaluation Sampling Time:
    config.sample_rate = config.crazyswarm_rate
    # Spline Parameters:
    config.spline_resolution = 7
    config.bin_resolution = 51
    return config

def main(argv=None):
    # Create Config Dict:
    params = get_config()

    # Create a block diagram containing our system.
    builder = DiagramBuilder()

    # Reference Motion:
    driver_reference = reference_trajectory.ReferenceTrajectory(config=params)
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
    driver_adversary = adversary_controller.Adversary(config=params)
    adversary = builder.AddSystem(driver_adversary)

    # Risk Learning:
    track_radius = 0.6
    driver_tracking_learner = learning_framework.RiskLearning(config=params, failure_radius=track_radius)
    driver_tracking_learner.evaluate = evaluator_extension.tracking_evaluation
    tracking_learner = builder.AddSystem(driver_tracking_learner)

    avoid_radius = 0.5
    driver_avoidance_learner = learning_framework.RiskLearning(config=params, failure_radius=avoid_radius)
    driver_avoidance_learner.evaluate = evaluator_extension.avoidance_evaluation
    avoidance_learner = builder.AddSystem(driver_avoidance_learner)

    # Connect Systems:
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

    # Adversary -> Motion Planner Obstacle Initial Condition
    builder.Connect(
        adversary.get_output_port(0),
        planner.get_input_port(driver_planner.avoider_adversary_states_input)
    )

    builder.Connect(
        reference.get_output_port(driver_reference.figure_eight_output),
        planner.get_input_port(driver_planner.tracker_adversary_states_input)
    )

    # Crazyswarm Out -> Learning Framework
    builder.Connect(
        crazyswarm.get_output_port(0),
        tracking_learner.get_input_port(driver_tracking_learner.agent_input)
    )

    builder.Connect(
        crazyswarm.get_output_port(0),
        avoidance_learner.get_input_port(driver_avoidance_learner.agent_input)
    )

    # Crazyswarm Out -> Adversary
    builder.Connect(
        crazyswarm.get_output_port(0),
        adversary.get_input_port(0)
    )

    # Adversary Out -> Learning Framework
    builder.Connect(
        adversary.get_output_port(0),
        avoidance_learner.get_input_port(driver_avoidance_learner.adversary_input)
    )

    builder.Connect(
        reference.get_output_port(driver_reference.figure_eight_output),
        tracking_learner.get_input_port(driver_tracking_learner.adversary_input)
    )

    # Learning Framework -> Motion Planner Constraints:
    builder.Connect(
        tracking_learner.get_output_port(0),
        planner.get_input_port(driver_planner.tracking_constraint_input)
    )

    builder.Connect(
        avoidance_learner.get_output_port(0),
        planner.get_input_port(driver_planner.avoidance_constraint_input)
    )

    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()

    # Create the simulator:
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)

    # Initialize Crazyswarm Objects:
    driver_crazyswarm.initialize_driver()
    driver_adversary.initialize_driver()

    # Initialize Simulator:
    simulator.Initialize()

    # Simulate System:
    FINAL_TIME = 40.0
    dt = 0.1
    next_time_step = dt

    # Lists to store data for animation:
    realtime_rate_history = []
    motion_planner_history = []
    parser_history = []
    adversary_history = []
    tracking_history = []
    tracker_fp_history = []
    tracker_ls_history = []
    tracker_data_history = []
    avoidance_fp_history = []
    avoidance_ls_history = []
    avoidance_data_history = []
    solve_time = []

    # w/ while-loop:
    while next_time_step <= FINAL_TIME:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
        realtime_rate_history.append(simulator.get_actual_realtime_rate())

        # Drone Control History:
        motion_planner_history.append(driver_planner._full_state_trajectory)
        parser_history.append(driver_parser._current_trajectory)

        # Desired Tracking History:
        tracking_history.append(driver_reference._figure_eight_reference[:2])

        # Chaser Drone Control History:
        adversary_history.append(driver_adversary._state_output)

        # Tracking Risk Learning Task:
        tracker_fp_history.append(driver_tracking_learner._fp_sol)
        tracker_ls_history.append(driver_tracking_learner._ls_sol)
        tracker_data_history.append(driver_tracking_learner._data)

        # Avoidance Risk Learning Task:
        avoidance_fp_history.append(driver_avoidance_learner._fp_sol)
        avoidance_ls_history.append(driver_avoidance_learner._ls_sol)
        avoidance_data_history.append(driver_avoidance_learner._data)

        # Solve Time History:
        solve_time.append(
            [
                driver_planner._optimizer_time,
                driver_tracking_learner.fpn.run_time,
                driver_tracking_learner.lsn.run_time,
                driver_avoidance_learner.fpn.run_time,
                driver_avoidance_learner.lsn.run_time,
            ]
        )
        try:
            simulator.AdvanceTo(next_time_step)
            next_time_step += dt
        except:
            print(f"Exception Occurred...")
            driver_crazyswarm.execute_landing_sequence()
            break

    # Land the Drone:
    driver_crazyswarm.execute_landing_sequence()

    # Plot results:
    fig_playback, ax_playback = plt.subplots()
    fig_playback.tight_layout(pad=2.5)
    planner_plot, = ax_playback.plot([], [], color='cornflowerblue', alpha=0.5, linewidth=1.0)
    node_plot, = ax_playback.plot([], [], color='cornflowerblue', marker='.', linewidth=0.5, linestyle='None')
    # Simulation Plot:
    ax_playback.axis('equal')
    ax_playback.set(xlim=(-5, 5), ylim=(-5, 5))
    ax_playback.set_xlabel('X')  # X Label
    ax_playback.set_ylabel('Y')  # Y Label

    # Animation
    ax_playback.set_title('Risk Learning Animation:')
    video_title = "hardware_risk_learning_playback"

    # Initialize Patches and Plots:
    adv = Circle((0, 0), radius=0.01, color='red')
    agn = Circle((0, 0), radius=0.01, color='cornflowerblue')
    ref = Circle((0, 0), radius=0.01, color='grey')
    ref_rad = Circle((0, 0), radius=track_radius, color='grey', alpha=0.1)
    adv_rad = Circle((0, 0), radius=avoid_radius, color='red', alpha=0.1)
    arena = Rectangle(
        (-params.area_bounds, -params.area_bounds),
        width=2*params.area_bounds,
        height=2*params.area_bounds,
        linewidth=1.0,
        edgecolor='red',
        facecolor='None',
        alpha=0.1,
    )
    ax_playback.add_patch(adv)
    ax_playback.add_patch(agn)
    ax_playback.add_patch(ref)
    ax_playback.add_patch(ref_rad)
    ax_playback.add_patch(adv_rad)
    ax_playback.add_patch(arena)

    # Setup Animation Writer:
    dpi = 300
    FPS = 20
    simulation_size = len(realtime_rate_history)
    sample_rate = int(1 / (dt * FPS))
    writerObj = FFMpegWriter(fps=FPS)

    # Plot and Create Animation:
    with writerObj.saving(fig_playback, video_title+".mp4", dpi):
        for i in range(0, simulation_size):
            dt_realtime = dt * realtime_rate_history[i]
            sample_rate = np.ceil(dt_realtime * FPS)

            if sample_rate <= 1:
                sample_rate = 1

            for _ in range(0, int(sample_rate)):
                # Plot Adversary:
                adv.center = adversary_history[i][0], adversary_history[i][1]
                adv_rad.center = adversary_history[i][0], adversary_history[i][1]
                # Reference:
                ref.center = tracking_history[i][0], tracking_history[i][1]
                ref_rad.center = tracking_history[i][0], tracking_history[i][1]
                # Plot Agent and Motion Planner Trajectory:
                position = parser_history[i]
                motion_plan = np.reshape(motion_planner_history[i], (6, -1))
                agn.center = position[0], position[1]
                planner_plot.set_data(motion_plan[0, :], motion_plan[1, :])
                node_plot.set_data(motion_plan[0, :], motion_plan[1, :])
                # Update Drawing:
                fig_playback.canvas.draw()  # Update the figure with the new changes
                # Grab and Save Frame:
                writerObj.grab_frame()

    # # Learning Framework:
    # fig_regression, ax_regression = plt.subplots(2)
    # fig_regression.tight_layout(pad=2.5)

    # data_plot, = ax_regression[0].plot([], [], color='black', marker='.', linestyle='None')
    # fp, = ax_regression[0].plot([], [], color='red')
    # ls, = ax_regression[1].plot([], [], color='red')

    # # FP Plot:
    # ax_regression[0].set_xlim([-1, 3])  # X Lim
    # ax_regression[0].set_ylim([-1, 2])  # Y Lim
    # ax_regression[0].set_xlabel('delta')  # X Label
    # ax_regression[0].set_ylabel('r(delta)')  # Y Label
    # # LS Plot:
    # ax_regression[1].set_xlim([-1, 3])  # X Lim
    # ax_regression[1].set_ylim([-2, 1])  # Y Lim
    # ax_regression[1].set_xlabel('delta')  # X Label
    # ax_regression[1].set_ylabel('s(delta)')  # Y Label

    # # Animation
    # ax_regression[0].set_title('Risk Learning Framework:')
    # video_title = "hardware_risk_learning_regression"

    # # Setup Animation Writer:
    # dpi = 300
    # FPS = 20
    # simulation_size = len(realtime_rate_history)
    # sample_rate = int(1 / (dt * FPS))
    # writerObj = FFMpegWriter(fps=FPS)

    # # Plot and Create Animation:
    # with writerObj.saving(fig_regression, video_title+".mp4", dpi):
    #     for i in range(0, simulation_size):
    #         dt_realtime = dt * realtime_rate_history[i]
    #         sample_rate = np.ceil(dt_realtime * FPS)

    #         if sample_rate <= 1:
    #             sample_rate = 1

    #         for _ in range(0, int(sample_rate)):
    #             # Plot FP:
    #             data_plot.set_data(data_history[i][0, :], data_history[i][1, :])
    #             fp.set_data(fp_history[i][0, :], fp_history[i][1, :])
    #             # Plot LS:
    #             ls.set_data(ls_history[i][0, :], ls_history[i][1, :])
    #             # Update Drawing:
    #             fig_regression.canvas.draw()  # Update the figure with the new changes
    #             # Grab and Save Frame:
    #             writerObj.grab_frame()


if __name__ == "__main__":
    main()