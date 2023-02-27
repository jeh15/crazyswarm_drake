import argparse
import shelve

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    args = parser.parse_args()

    db = shelve.open(args.filename)

    fig, ax = plt.subplots(2)
    fig.tight_layout(pad=2.5)

    data, = ax[0].plot([], [], color='black', marker='.', linestyle='None')
    fp, = ax[0].plot([], [], color='red')
    ls, = ax[1].plot([], [], color='red')
    dh, = ax[1].plot([], [], color='black', marker='.', linestyle='None')

    # FP Plot:
    ax[0].set_xlim([-1, 3])  # X Lim
    ax[0].set_ylim([-1, 2])  # Y Lim
    ax[0].set_xlabel('delta')  # X Label
    ax[0].set_ylabel('r(delta)')  # Y Label
    # LS Plot:
    ax[1].set_xlim([-1, 3])  # X Lim
    ax[1].set_ylim([-2, 1])  # Y Lim
    ax[1].set_xlabel('delta')  # X Label
    ax[1].set_ylabel('s(delta)')  # Y Label

    # Animation
    ax[0].set_title('Learning Playback:')
    video_title = args.filename.split("shelve")[0] + "_learning"

    # Setup Animation Writer:
    dpi = 300
    FPS = 20
    simulation_size = len(db['realtime_rate_history'])
    writerObj = FFMpegWriter(fps=FPS)

    # Plot and Create Animation:
    with writerObj.saving(fig, video_title+".mp4", dpi):
        for i in range(0, simulation_size):
            # Plot FP:
            data.set_data(db['raw_data_history'][i][0, :], db['raw_data_history'][i][1, :])
            fp.set_data(db['failure_probability_history'][i][0, :], db['failure_probability_history'][i][1, :])
            # Plot LS:
            ls.set_data(db['log_survival_history'][i][0, :], db['log_survival_history'][i][1, :])
            # Plot Delta:
            dh.set_data(db['delta_history'][i], db['risk_history'][i])
            # Update Drawing:
            fig.canvas.draw()  # Update the figure with the new changes
            # Grab and Save Frame:
            writerObj.grab_frame()
    
    db.close()


if __name__ == "__main__":
    main()
