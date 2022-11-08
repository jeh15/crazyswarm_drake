import matplotlib.pyplot as plt
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import LogVectorOutput

import crazyswarm_class

# Create a simple block diagram containing our system.
builder = DiagramBuilder()
system = builder.AddSystem(crazyswarm_class.CrazyswarmSystem())
logger = LogVectorOutput(system.get_output_port(0), builder)
diagram = builder.Build()

# Set the initial conditions, x(0).
context = diagram.CreateDefaultContext()

# Create the simulator, and simulate for 1 seconds.
simulator = Simulator(diagram, context)
simulator.Initialize()
simulator.AdvanceTo(1)

# Plot the results.
log = logger.FindLog(context)
plt.figure()
plt.plot(log.sample_times(), log.data().transpose())
plt.xlabel('t')
plt.ylabel('x(t)')
plt.show()

plt.savefig('foo.png')
print(log.data().transpose())