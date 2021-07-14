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
    session_spikes_arr: np.ndarray,
    session_data: dict,
    granularity: str,
    smooth: bool,
    **kwargs,
):
    """
    Plots the neuron firing rate.

        Input:
            - session_spikes_arr: the spikes data is fed separately if
              it is preprocessed (eg. brain regions/areas).
            - session_data: single session data subset from all_data.
            - granularity: "session" or "trial".
            - smooth: a bool value to apply low pass filter.
            - if granularity="trial", then it requires a new variable called
              trial_id.
        Output:
            - Line plot.
    """

    # time axis
    dt = session_data["bin_size"]
    T = session_data["spks"].shape[-1]
    time_steps = dt * np.arange(T)

    if granularity == "session":
        session_spikes = session_spikes_arr

        # events
        response = session_data["response"]
        stim_onset = session_data["stim_onset"]

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
        plt.axvline(x=stim_onset, color="red", label="Stimulus Onset")
        plt.title(f"Average Neuron Firing Rate Over A Single Session")
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Firing Rate (Hz)")
        plt.show()

    elif granularity == "trial":
        trial_id = kwargs["trial_id"]
        trial_spikes = session_spikes_arr[:, trial_id, :]

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
            f"Average Firing Rate Over A Single Trial (response = {idx2response[response]}, feedback={idx2feedback[feedback]})"
        )
        plt.legend()
        plt.xlabel("Time (seconds)")
        plt.ylabel("Firing Rate (Hz)")
        plt.show()


def plot_spikes_raster(trial_spikes_arr: np.ndarray, cmap: str = "gray_r"):
    """
    Plots raster visualization of spiking data. Uses heatmap under the hood
    because of the preprocessed data.
    """
    plt.figure(figsize=(7, 7))
    sns.heatmap(trial_spikes_arr, cmap=cmap)
    plt.title("Spiking Activity of Neurons In A Single Trial")
    plt.xlabel("Time (binned by 10 msec)")
    plt.ylabel("Individual Neurons")
    plt.show()
