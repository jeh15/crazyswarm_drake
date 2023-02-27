import argparse
import shelve

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle

import ml_collections


def get_config() -> ml_collections.ConfigDict():
    config = ml_collections.ConfigDict()
    config.area_bounds = 1.0
    config.failure_radius = 0.5
    return config


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    params = get_config()

    db = shelve.open(args.filename)

    fig, ax = plt.subplots()
    fig.tight_layout(pad=2.5)
    p, = ax.plot([], [], color='cornflowerblue', alpha=0.5, linewidth=1.0)
    
    # Simulation Plot:
    ax.axis('equal')
    ax.set(xlim=(-5, 5), ylim=(-5, 5))
    ax.set_xlabel('X')  # X Label
    ax.set_ylabel('Y')  # Y Label

    # Animation
    ax.set_title('Test Playback:')
    video_title = args.filename.split("shelve")[0] + "test"

    # Initialize Patches and Plots:
    adv = Circle((0, 0), radius=0.01, color='red')
    agn = Circle((0, 0), radius=0.01, color='cornflowerblue')
    rad = Circle((0, 0), radius=params.failure_radius, color='red', alpha=0.1)
    arena = Rectangle(
        (-params.area_bounds, -params.area_bounds),
        width=2*params.area_bounds,
        height=2*params.area_bounds,
        linewidth=1.0,
        edgecolor='red',
        facecolor='None',
        alpha=0.1,
    )
    ax.add_patch(adv)
    ax.add_patch(agn)
    ax.add_patch(rad)
    ax.add_patch(arena)

    # Setup Animation Writer:
    dpi = 300
    FPS = 20
    simulation_size = len(db['realtime_rate_history'])
    writerObj = FFMpegWriter(fps=FPS)

    # Plot and Create Animation:
    with writerObj.saving(fig, video_title + ".mp4", dpi):
        for i in range(0, simulation_size):
            # Plot Adversary:
            adv.center = db['adversary_state_history'][i][0], db['adversary_state_history'][i][1]
            rad.center = db['adversary_state_history'][i][0], db['adversary_state_history'][i][1]
            # Plot Agent and Motion Planner Trajectory:
            agn.center = db['agent_state_history'][i][0], db['agent_state_history'][i][1]
            motion_plan = np.reshape(db['motion_plan_history'][i], (6, -1))
            p.set_data(motion_plan[0, :], motion_plan[1, :])
            # Update Drawing:
            fig.canvas.draw()
            # Grab and Save Frame:
            writerObj.grab_frame()
    
    db.close()


if __name__ == "__main__":
    main()
