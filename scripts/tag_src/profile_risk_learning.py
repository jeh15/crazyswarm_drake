import argparse
import cProfile as profile
from contextlib import contextmanager
import pstats
import subprocess
import signal
import os
import timeit

import numpy as np
import ml_collections

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource

# Custom LeafSystems:
import risk_learning_module as rlm

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

@contextmanager
def use_py_spy(output_file, *, native=True, sudo=True):
    """Use py-spy in specific context."""
    args = [
        "py-spy",
        "record",
        "-o", output_file,
        "--pid", str(os.getpid()),
    ]
    if native:
        # This will include C++ symbols as well, which will help determine if
        # there which C++ functions may be slowing things down. However, you
        # will need to dig or Ctrl+F to find out what Python code is going
        # slow. Using Ctrl+F, searching for ".py:" should highlight Python
        # code.
        args += ["--native"]
    if sudo:
        args = ["sudo"] + args
    p = subprocess.Popen(args)
    try:
        yield
    finally:
        p.send_signal(signal.SIGINT)
        p.wait()


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
    # Parse Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cprofile", type=str)
    parser.add_argument("--py_spy", type=str)
    parser.add_argument(
        "--py_spy_no_native",
        dest="py_spy_native",
        action="store_false",
    )
    args = parser.parse_args()

    # Create Config Dict:
    params = get_config()

    # Create a block diagram containing our system.
    builder = DiagramBuilder()

    # Risk Learning: Failure Learning
    driver_risk = rlm.RiskLearning(config=params)
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
    simulator.set_target_realtime_rate(0.0)
    simulator.Initialize()

    # Simulate System:
    dt = 0.1
    next_time_step = dt
    benchmark_time = 20.0
    wall_t_end = 1.0

    if args.py_spy is not None:
        with use_py_spy(args.py_spy, native=args.py_spy_native):
            wall_t_start = timeit.default_timer()
            while next_time_step <= benchmark_time:
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
            wall_t_end = timeit.default_timer() - wall_t_start
    if args.cprofile is not None:
        with use_cprofile(args.cprofile):
            wall_t_start = timeit.default_timer()
            while next_time_step <= benchmark_time:
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
            wall_t_end = timeit.default_timer() - wall_t_start

    realtime_factor = benchmark_time / wall_t_end
    print(f"realtime factor: {realtime_factor:.3f}")


if __name__ == "__main__":
    main()
