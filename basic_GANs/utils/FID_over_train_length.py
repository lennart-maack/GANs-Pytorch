import os
import pandas as pd
from matplotlib import pyplot as plt


def create_x_values(result) -> list:

    x_values = []
    for i in range(len(result)):
        x_values.append(i*600)

    return x_values


def create_FID_over_nkimgs_plot(training_session_dir, output_dir, name):

    """
    Creates a .png file with the FID score plotted over the number of kimgs seen while training.

    training_session_dir: Directory to the folder of one or multiple training sessions - each training session has! to have a metrics.csv included which has as column 'fid50k_full'

    output_dir: Directory where the png file should be saved

    name: to name the .png file uniquely

    """

    frames = []
    for _, dirs, _ in os.walk(training_session_dir):
        for subdir in dirs:
                if os.path.exists(os.path.join(training_session_dir, subdir, "metrics.csv")):
                    frames.append(pd.read_csv(os.path.join(training_session_dir, subdir, "metrics.csv")))

    result = pd.concat(frames)

    x_values = create_x_values(result)

    y_values = result["fid50k_full"]

    fig = plt.figure()
    ax=fig.add_subplot()
    ax.plot(x_values, y_values, color="C0")

    ax.set(xlabel="Number k imgs", ylabel="FID", title=f"FID over kimgs seen by GAN {name}")
    ax.set_ylim([10, 500])
    ax.yaxis.set_ticks([25, 50, 75, 100, 150, 200, 300, 450])

    ax.grid()

    plt.savefig(os.path.join(output_dir, f"FID_over_nkimgs_{name}.png"))




if __name__ == "__main__":
    
    name = "DiffAUG_2"
    training_session_dir = r"G:\My Drive\Projektarbeit_ResearchProject\experiments\IVUS\IVUS_STYLEGAN2_DIFFAUG\IVUS_11_20_2"

    output_dir = r"G:\My Drive\Projektarbeit_ResearchProject\experiments\IVUS\FID_over_nkimgs\DiffAUG"
    
    create_FID_over_nkimgs_plot(training_session_dir, output_dir, name)