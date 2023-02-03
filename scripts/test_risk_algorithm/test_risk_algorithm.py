import argparse
import numpy as np
import ml_collections

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

import matplotlib.pyplot as plt
import pdb

# Custom LeafSystems:
import risk_algorithm_new as ran


# Random Walk:
def random_walk(current_position: np.ndarray, target_position: np.ndarray, dt: float) -> np.ndarray:
    # Find direction to target:
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
    FINAL_TIME = 10.0
    dt = 0.1
    next_time_step = dt

    while next_time_step <= FINAL_TIME:
        print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
        simulator.AdvanceTo(next_time_step)
        # Get Subsystem Context for ConstantSourceVector:
        agent_context = driver_agent.GetMyContextFromRoot(context)
        agent_position = driver_agent.get_mutable_source_value(agent_context)
        adversary_context = driver_adversary.GetMyContextFromRoot(context)
        adversary_position = driver_adversary.get_source_value(adversary_context)
        # Random walk towards stationary adversary: TODO(jeh15) Add Reset Event
        new_position = random_walk(
            current_position=np.asarray(agent_position.value()),
            target_position=np.asarray(adversary_position.value()),
            dt=dt
        )
        agent_position.set_value(new_position)
        # Increment time step:
        next_time_step += dt

    pdb.set_trace()


if __name__ == "__main__":
    main()
