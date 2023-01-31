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

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import pdb

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# Custom LeafSystems:
import motion_planner
import reference_trajectory
import trajectory_parser
import crazyswarm_class
import adversary_tracker

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

# Convenient Data Class:
def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    # Control Rates:
    config.motion_planner_rate = 1.0 / 50.0
    config.reference_trajectory_rate = 1.0 / 50.0
    config.crazyswarm_rate = 1.0 / 200.0
    # Model Parameters:
    config.nodes = 51                   # (Discretized Points)
    config.control_horizon = 11         # (Node to control to) Not used
    config.state_dimension = 2          # (x, y)
    config.time_horizon = 2.0
    config.dt = config.time_horizon / (config.nodes - 1.0)
    return config

def main(argv=None):
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

    # Adversary Tracker:
    driver_adversary = adversary_tracker.Adversary(config=params)
    adversary = builder.AddSystem(driver_adversary)

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
    diagram = builder.Build()

    # Set the initial conditions, x(0).
    context = diagram.CreateDefaultContext()

    # Create the simulator:
    simulator = Simulator(diagram, context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

   # Simulate System:
    benchmark_time = 20.0
    wall_t_end = 1.0

    if args.py_spy is not None:
        with use_py_spy(args.py_spy, native=args.py_spy_native):
            wall_t_start = timeit.default_timer()
            simulator.AdvanceTo(benchmark_time)
            wall_t_end = timeit.default_timer() - wall_t_start
    if args.cprofile is not None:
        with use_cprofile(args.cprofile):
            wall_t_start = timeit.default_timer()
            simulator.AdvanceTo(benchmark_time)
            wall_t_end = timeit.default_timer() - wall_t_start
    
    driver_crazyswarm.execute_landing_sequence()
    
    realtime_factor = benchmark_time / wall_t_end     
    print(f"realtime factor: {realtime_factor:.3f}")

    # w/o while-loop:
    # simulator.AdvanceTo(FINAL_TIME)
    # print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")

    # Land the Drone:
    # driver_crazyswarm.execute_landing_sequence()


if __name__ == "__main__":
    main()