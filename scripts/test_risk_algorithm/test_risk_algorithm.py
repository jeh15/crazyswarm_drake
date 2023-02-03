import argparse
import numpy as np
import ml_collections

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import pdb

# Custom LeafSystems:
import risk_algorithm_new as ran


# Random Walk:
def directed_walk(
    current_position: np.ndarray,
    target_position: np.ndarray, 
    dt: float, 
    failure_radius: float
) -> np.ndarray:
    # Find direction to target:
    direction = target_position[:2] - current_position[:2]
    # Check reset condition:
    if (np.linalg.norm(direction) - failure_radius <= 0):
        current_position = np.random.uniform(low=-1.0, high=1.0, size=(2,))
        direction = target_position[:2] - current_position[:2]
    # Normalize direction vector:
    direction /= np.max(np.abs(direction), axis=0)
    # Create noise vector to inject into the direction:
    noise = (np.random.rand(2,) - 0.5)
    new_position = current_position[:2] + (direction + noise) * dt
    # Fill out other indices:
    zeros = np.zeros((7,))
    return np.concatenate([new_position, zeros])


# Convenient Data Class:
def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    # Control Rates:
    config.motion_planner_rate = 1.0 / 10.0
    config.reference_trajectory_rate = 1.0 / 50.0
    config.crazyswarm_rate = 1.0 / 100.0
    # Model Parameters:
    config.nodes = 21                   # (Discretized Points)
    config.control_horizon = 11         # (Node to control to) Not used
    config.state_dimension = 2          # (x, y)
    config.time_horizon = 2.0
    config.dt = config.time_horizon / (config.nodes - 1.0)
    # Spline Parameters:
    config.spline_resolution = 4
    config.bin_resolution = 100
    config.failure_radius = 0.1
    return config


def main(argv=None):
    # Create Config Dict:
    params = get_config()

    # Create a block diagram containing our system.
    builder = DiagramBuilder()

    # Risk Learning: Failure Learning
    driver_risk = ran.RiskAlgorithm(config=params)
    risk = builder.AddSystem(driver_risk)

    # Dummy Agent:
    driver_agent = ConstantVectorSource(np.ones((9,), dtype=float))
    agent = builder.AddSystem(driver_agent)

    # Dummy Adversary:
    driver_adversary = ConstantVectorSource(np.zeros((6,), dtype=float))
    adversary = builder.AddSystem(driver_adversary)

    # Connect Systems:
    # Agent Out -> Risk Learning
    builder.Connect(
        agent.get_output_port(0),
        risk.get_input_port(driver_risk.agent_input)
    )

    # Adversary Out -> Risk Learning
    builder.Connect(
        adversary.get_output_port(0),
        risk.get_input_port(driver_risk.adversary_input)
    )

    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()

    # Create the simulator:
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Simulate System:
    FINAL_TIME = 15.0
    dt = 0.1
    next_time_step = dt

    fp_history = []
    ls_history = []
    agent_history = []

    while next_time_step <= FINAL_TIME:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
        simulator.AdvanceTo(next_time_step)
        # Get Subsystem Context for ConstantSourceVector:
        agent_context = driver_agent.GetMyContextFromRoot(context)
        agent_position = driver_agent.get_mutable_source_value(agent_context)
        adversary_context = driver_adversary.GetMyContextFromRoot(context)
        adversary_position = driver_adversary.get_source_value(adversary_context)
        # Random walk towards stationary adversary: TODO(jeh15) Add Reset Event
        new_position = directed_walk(
            current_position=np.asarray(agent_position.value()),
            target_position=np.asarray(adversary_position.value()),
            dt=dt,
            failure_radius=params.failure_radius,
        )
        agent_position.set_value(new_position)
        # Increment time step:
        next_time_step += dt
        # Save data for plotting:
        fp_history.append(driver_risk._fp_sol)
        ls_history.append(driver_risk._ls_sol)
        agent_history.append(new_position)

    # Plot results:
    fig, ax = plt.subplots(3)
    sim, = ax[0].plot([], [], color='red')
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
    video_title = "simulation_risk_learning"

    # Initialize Patch:
    adv = Circle((0, 0), radius=0.01, color='red')
    agn = Circle((0, 0), radius=0.01, color='cornflowerblue')
    rad = Circle((0, 0), radius=params.failure_radius, color='red', alpha=0.1)
    ax[0].add_patch(adv)
    ax[0].add_patch(agn)
    ax[0].add_patch(rad)

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
            # Plot Agn Trajectory:
            agn.center = agent_history[i][0], agent_history[i][1]
            # Plot FP:
            fp.set_data(fp_history[i][0, :], fp_history[i][1, :])
            # Plot LS:
            ls.set_data(ls_history[i][0, :], ls_history[i][1, :])
            # Update Drawing:
            fig.canvas.draw()  # Update the figure with the new changes
            # Grab and Save Frame:
            writerObj.grab_frame()

if __name__ == "__main__":
    main()
