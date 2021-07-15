import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

sns.set_style("darkgrid")


def get_firing_rate(spikes_arr: np.ndarray, bin_size: float) -> np.ndarray:
    """
    Returns firing rate of neurons. `bin_size` is predefined
    in the data (`all_data[session_id]["bin_size"]`).
    """
    return (1 / bin_size) * spikes_arr


def smoothen_firing_rate(
    firing_rate: np.ndarray,
    order: int = 3,
    wn: int = 4000,
    btype: int = "lowpass",
    fs: int = 50000,
) -> np.ndarray:
    """
    Applies smoothening filter on firing rate.
    """
    b, a = butter(order, wn, btype, fs=fs)

    return filtfilt(b, a, firing_rate)


def plot_firing_rate(
    spikes_arr: np.ndarray,
    session_data: dict,
    granularity: str,
    smooth: bool,
    trial_id: int = None,
    title_info: dict = None,
):
    """
    Plots the neuron firing rate.

        Input:
            - spikes_arr: the spikes data is fed separately if
              it is preprocessed (eg. brain regions/areas). It is
              3-d array if is spikes from the whole session or it is a
              2-d array if it is spikes from a single trial.
            - session_data: single session data subset from all_data.
            - granularity: "session" or "trial".
            - smooth: a bool value to apply low pass filter.
            - trial_id: required if granularity="trial".
            - title_info: a dictionary with 2 keys. Example = {"session_id": 1, "trial_id": 1}
        Output:
            - Line plot.
    """

    # time axis
    dt = session_data["bin_size"]
    T = session_data["spks"].shape[-1]
    time_steps = dt * np.arange(T)

    if granularity == "session":
        session_spikes = spikes_arr

        # events
        response = session_data["response"]

        # base firing rate
        firing_rate = get_firing_rate(spikes_arr=session_spikes, bin_size=dt)

        # event firing rates
        firing_rate_left = firing_rate[:, response == 1].mean(axis=(0, 1))
        firing_rate_center = firing_rate[:, response == 0].mean(axis=(0, 1))
        firing_rate_right = firing_rate[:, response == -1].mean(axis=(0, 1))

        if smooth:
            firing_rate_left = smoothen_firing_rate(firing_rate_left)
            firing_rate_center = smoothen_firing_rate(firing_rate_center)
            firing_rate_right = smoothen_firing_rate(firing_rate_right)

        # plot figure
        plt.figure(figsize=(15, 5))
        plt.plot(time_steps, firing_rate_left, label="Left Turn")
        plt.plot(time_steps, firing_rate_center, label="Center Stay")
        plt.plot(time_steps, firing_rate_right, label="Right Turn")
        plt.axvline(x=session_data["stim_onset"], color="red", label="Stimulus Onset")
        if title_info:
            plt.title(
                f"Average Neuron Firing Rate for {' '.join([f'{k}={v}' for k,v in title_info.items()])}"
            )
        else:
            plt.title(f"Average Neuron Firing Rate For a Single Session")
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Firing Rate (Hz)")
        plt.show()

    elif granularity == "trial":
        # if spikes_arr is already filtered for a trial.
        # This may happen if it's filtered by brain area/
        # region.

        if len(spikes_arr.shape) == 2:
            trial_spikes = spikes_arr
        else:
            trial_spikes = spikes_arr[:, trial_id, :]

        # event
        response = session_data["response"][trial_id]
        feedback = session_data["feedback_type"][trial_id]

        # base firing rate
        firing_rate = get_firing_rate(spikes_arr=trial_spikes, bin_size=dt).mean(axis=0)

        if smooth:
            firing_rate = smoothen_firing_rate(firing_rate)

        # for plot title
        idx2response = {-1: "right", 0: "center", 1: "left"}
        idx2feedback = {1: "positive", -1: "negative"}

        plt.figure(figsize=(15, 5))
        plt.plot(dt * np.arange(T), firing_rate)
        plt.axvline(x=session_data["stim_onset"], color="red", label="Stimulus Onset")
        plt.axvline(x=session_data["gocue"][trial_id], color="green", label="Go Cue")
        plt.axvline(
            x=session_data["response_time"][trial_id],
            color="orange",
            label="Response Time",
        )
        plt.axvline(
            x=session_data["feedback_time"][trial_id],
            color="purple",
            label="Feedback Time",
        )
        plt.title(
            f"Average Neuron Firing Rate for {' '.join([f'{k}={v}' for k,v in title_info.items()])} with (response = {idx2response[response]}, feedback={idx2feedback[feedback]})"
        )
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Firing Rate (Hz)")
        plt.show()


def plot_spikes_raster(
    trial_spikes_arr: np.ndarray, title_info: dict = None, cmap: str = "gray_r"
):
    """
    Plots raster visualization of spiking data. Uses heatmap under the hood
    because of the preprocessed data.

        Inputs
            - trial_spikes_arr: a 2-d numpy array with spiking data as 1/0.
            - title_info: a dictionary with 2 keys. Example = {"Trial": 1, "Session": 1}.
            - cmap: colormap.

        Output:
            - raster plot built using seaborn heatmap.
    """
    plt.figure(figsize=(7, 7))
    sns.heatmap(trial_spikes_arr, cmap=cmap)
    if title_info:
        plt.title(
            f"Spiking Activity in {' of '.join([f'{k}={v}' for k,v in title_info.items()])}"
        )
    else:
        plt.title("Spiking Activity in a Single Trial")
    plt.xlabel("Time (binned by 10 msec)")
    plt.ylabel("Individual Neurons")
    plt.show()
