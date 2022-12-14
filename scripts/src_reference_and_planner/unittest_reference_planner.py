import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import numpy as np

from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput, ConstantVectorSource

import pdb

# Custom LeafSystems:
import motion_planner
import reference_trajectory

# Create a block diagram containing our system.
builder = DiagramBuilder()

# Reference Motion:
driver_reference = reference_trajectory.FigureEight()
reference = builder.AddSystem(driver_reference)

# Motion Planner:
driver_planner = motion_planner.QuadraticProgram()
planner = builder.AddSystem(driver_planner)

# Create Dummy Inputs:
driver_ic = ConstantVectorSource(np.zeros((6,), dtype=float))
dummy_initial_condition = builder.AddSystem(driver_ic)

# Connect Systems:
builder.Connect(
    reference.get_output_port(0),
    planner.get_input_port(driver_planner.target_input)
    )

builder.Connect(
    dummy_initial_condition.get_output_port(0),
    planner.get_input_port(driver_planner.initial_condition_input)
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

trajectory_history = []
ic_history = []

while next_time_step <= FINAL_TIME:
    print(f"Drake Real Time Rate: {simulator.get_actual_realtime_rate()}")
    simulator.AdvanceTo(next_time_step)
    # Get Subsystem Context for ConstantSourceVector:
    subsystem_context = driver_ic.GetMyContextFromRoot(context)
    src_value = driver_ic.get_mutable_source_value(subsystem_context)
    # Get next IC: (Pretend it controls only first 3 Nodes)
    trajectory = np.reshape(
        driver_planner._full_state_trajectory,
        (-1, driver_planner._num_nodes)
        )
    trajectory_history.append(trajectory[:, :3])
    new_ic = trajectory[:, 2]
    ic_history.append(new_ic)
    src_value.set_value(new_ic)

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
ax.add_patch(k)

# Setup Animation Writer:
dpi = 300
FPS = 20
simulation_size = len(log_planner.sample_times())
dummy_size = len(ic_history[:])
dt = FINAL_TIME / dummy_size
sample_rate = int(1 / (dt * FPS))
writerObj = FFMpegWriter(fps=FPS)

# Resampled based on dummy size:
idx = np.round(np.linspace(0, simulation_size - 1, dummy_size)).astype(int)
log_planner_position = log_planner.data()[:, idx]
log_reference_position = log_reference.data()[:, idx]

# Plot and Create Animation:
with writerObj.saving(fig, video_title+".mp4", dpi):
    for i in range(0, dummy_size, sample_rate):
        # Plot Reference Trajectory:
        r.center = log_reference_position[0, i], log_reference_position[1, i]
        # Update Patch:
        c.center = log_planner_position[0, i], log_planner_position[1, i]
        k.center = ic_history[i][0],  ic_history[i][1]
        # Update Drawing:
        fig.canvas.draw()  # Update the figure with the new changes
        # Grab and Save Frame:
        writerObj.grab_frame()
