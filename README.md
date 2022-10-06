# Crazyflie Drone implementation using Crazyswarm and Pydrake

## Setup Note:

* Refer to [Pydrake Installation](https://drake.mit.edu/installation.html) to find whats compatable with your system.

| Operating System                  | Architecture  | Python    |
| -------------                     | ------------- | ----------|
| Ubuntu 20.04 LTS (Focal Fossa)    | x86_64        | 3.8       |
| Ubuntu 22.04 LTS (Jammy Jellyfish)| x86_64        | 3.10      |
| macOS Big Sur (11)                | x86_64        |	3.10    |
| macOS Monterey (12)               |	x86_64 or arm64 | 3.10  |

* Installing Pydrake via pip: [Drake Installation Guide](https://drake.mit.edu/pip.html#stable-releases)
    1. Create a virtual environment and install Drake:
    ```
    python3 -m venv env
    env/bin/pip install --upgrade pip
    env/bin/pip install drake
    ```
    2. Depending on your system you may need to install additional dependencies.
    3. Activate the virtual environment:
    ```
    source env/bin/activate
    ```
    (Note: Remember to source the virtual environment)