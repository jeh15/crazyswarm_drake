# Crazyflie Drone implementation using Crazyswarm and Pydrake

## Pre-requisite:
This Docker is built using ``nvidia/cuda`` as a base.

To use this you, follow these installation instructions:
* Download Docker and follow install instructions: https://docs.docker.com/desktop/install/linux-install/
    * Do not forget to add User to Docker group: sudo usermod -aG docker [computer-user-name]
    (You will need to restart for this to take effect: sudo reboot)
    
* Follow NVidia's instructions for nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    Do not forget pre-requisites: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html

## How to run:

Everything is contained in a Dockerfile. For easy use run ``./run_container.sh``
