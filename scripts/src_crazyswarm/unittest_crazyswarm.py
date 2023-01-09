import numpy as np
import ml_collections
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput

import pdb

# Custom LeafSystems:
import motion_planner
import reference_trajectory
import trajectory_parser
import crazyswarm_class


# Convenient Data Class:
def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    # Model Parameters:
    config.nodes = 21           # (Discretized Points)
    config.time_horizon = 1.0   # (Time Seconds)
    config.control_horizon = 3  # (Node to control to)
    config.state_dimension = 2  # (x, y)
    config.dt = config.time_horizon / (config.nodes - 1.0)
    # Control Rates:
    config.motion_planner_rate = 1.0 / 10.0
    config.reference_trajectory_rate = 1.0 / 10.0
    config.crazyswarm_rate = 1.0 / 100.0
    return config


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

# Logger:
logger_reference = LogVectorOutput(reference.get_output_port(0), builder)
logger_planner = LogVectorOutput(planner.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()

# Create the simulator:
simulator = Simulator(diagram, context)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

# Simulate System:
FINAL_TIME = 5.0
dt = 1.0 / 100.0
next_time_step = dt

while next_time_step <= FINAL_TIME:
    print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
    simulator.AdvanceTo(next_time_step)
    # pdb.set_trace()
    # Increment time step:
    next_time_step += dt

# Plot the results:
log_planner = logger_planner.FindLog(context)
log_reference = logger_reference.FindLog(context)


# Setup Figure: Initialize Figure / Axe Handles
fig, ax = plt.subplots()
p, = ax.plot([], [], color='red')
ax.axis('equal')
ax.set_xlim([-3, 3])  # X Lim
ax.set_ylim([-3, 3])  # Y Lim
ax.set_xlabel('X')  # X Label
ax.set_ylabel('Y')  # Y Label
ax.set_title('Reference + Planner Animation:')
video_title = "simulation"

# Initialize Patch:
c = Circle((0, 0), radius=0.1, color='cornflowerblue')
r = Circle((0, 0), radius=0.1, color='red')
k = Circle((0, 0), radius=0.1, color='black')
ax.add_patch(c)
ax.add_patch(r)

# Setup Animation Writer:
dpi = 300
FPS = 20
simulation_size = len(log_planner.sample_times())
dt = FINAL_TIME / simulation_size
sample_rate = int(1 / (dt * FPS))
writerObj = FFMpegWriter(fps=FPS)

# Resampled based on dummy size:
log_planner_position = log_planner.data()
log_reference_position = log_reference.data()

# Plot and Create Animation:
with writerObj.saving(fig, video_title+".mp4", dpi):
    for i in range(0, simulation_size, sample_rate):
        # Plot Reference Trajectory:
        r.center = log_reference_position[0, i], log_reference_position[1, i]
        # Update Patch:
        data = np.reshape(log_planner.data()[:, i], (6, -1))
        c.center = data[0, 0], data[1, 0]
        # Update Drawing:
        fig.canvas.draw()  # Update the figure with the new changes
        # Grab and Save Frame:
        writerObj.grab_frame()
