# Crazyflie Drone implementation using Crazyswarm and Pydrake
For simulations you can test your code via the Docker container. For hardware tests you will need to upload your code to the drone computer.

## Docker: Pre-requisites and Requirements:
This Docker is built using ``nvidia/cuda`` as a base. 

To use this you will need:

   ``GNU/Linux x86_64 with kernel version > 3.10``

   ``Docker >= 19.03`` (recommended, but some distributions may include older versions of Docker. The minimum supported version is 1.12)

   ``NVIDIA GPU with Architecture >= Kepler (or compute capability 3.0)``

   ``NVIDIA Linux drivers >= 418.81.07`` (Note that older driver releases or branches are unsupported.)

If you meet the above system requiremes use these following installation instructions:
* Download Docker and follow install instructions: https://docs.docker.com/desktop/install/linux-install/
    * Do not forget to add User to Docker group: sudo usermod -aG docker [computer-user-name]
    (You will need to restart for this to take effect: sudo reboot)
    
* Follow NVidia's instructions for nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    Do not forget pre-requisites: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html

## How to run Docker:

Simulation Environment is contained in a Dockerfile. To use simply run ``./run_container.sh``

## Developing:
Code Style Guide:

Use the same conventions as documented:
  * Python: https://google.github.io/styleguide/pyguide.html
  * C++:    https://google.github.io/styleguide/cppguide.html
  
### Naming Convention reference:
![Naming Reference](/imgs/naming_convention_reference.png)
