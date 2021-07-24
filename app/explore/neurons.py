"""
Functions to explore neuron specific data.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FactorAnalysis, FastICA

from app.utils.util import sort_neurons_by_brain_region, get_decision_type

sns.set_style("darkgrid")


def plot_spikes_raster(
    all_data: dict, session_id: int, trial_id: int, cmap: str = "gray_r"
):
    """Plots raster visualization of spiking data.

    Uses heatmap under the hood.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.
    trial_id: int
        Integer that denotes a trial within a session.
    cmap
        Plot colormap.
    """

    # extract session data.
    session_data = all_data[session_id]

    # important trial features.
    response = session_data["response"][trial_id]
    feedback = session_data["feedback_type"][trial_id]
    contrast_left = session_data["contrast_left"][trial_id]
    contrast_right = session_data["contrast_right"][trial_id]
    decision_type = get_decision_type(response=response, feedback=feedback)

    # get sorted spikes and filter out the required trial.
    idx_region_list, spikes_arr, sorted_neuron_idx = sort_neurons_by_brain_region(
        all_data=all_data, session_id=session_id
    )
    spikes_arr = spikes_arr[:, trial_id, :]

    # get important time events (vertical lines on the plot).
    stim_onset = session_data["stim_onset"] / session_data["bin_size"]
    gocue = session_data["gocue"][trial_id] / session_data["bin_size"]
    response_time = session_data["response_time"][trial_id] / session_data["bin_size"]
    feedback_time = session_data["feedback_time"][trial_id] / session_data["bin_size"]

    # plot
    plt.figure(figsize=(15, 10))
    sns.heatmap(spikes_arr, cmap=cmap, cbar_kws={"label": "Number of Spikes"})

    # add real neuron ids sorted by brain region
    # sns.heatmap(spikes_arr, yticklabels=sorted_neuron_idx, cmap=cmap, cbar_kws={"label": "Number of Spikes"})
    #  plt.yticks(fontsize=5)

    # for plot info
    idx2response = {-1: "right", 0: "center", 1: "left"}
    idx2feedback = {1: "positive", -1: "negative"}

    plt.suptitle(
        f"Spiking Activity in Session={session_id}  Trial={trial_id}  Decision Type={decision_type}",
        ha="center",
    )
    plt.title(
        f"response = {idx2response[response]}, feedback={idx2feedback[feedback]}, contrast-left={contrast_left}, contrast-right={contrast_right}",
        ha="center",
    )
    plt.xlabel("Time (binned by 10 msec)")
    plt.ylabel("Individual Neurons")

    # add horizontal lines for all brain regions
    for idx, region in idx_region_list:
        plt.axhline(y=idx, color="black")
        plt.text(2, idx - 4, region)

    # trial events
    plt.axvline(x=stim_onset, color="red", label="Stimulus Onset")
    plt.axvline(x=gocue, color="green", label="Go Cue")
    plt.axvline(x=response_time, color="orange", label="Response Time")
    plt.axvline(x=feedback_time, color="purple", label="Feedback Time")

    # add legend
    plt.legend(bbox_to_anchor=(1, 1), loc="lower left", ncol=1, fancybox=True)
    plt.show()


def get_firing_rate(spikes_arr: np.ndarray, bin_size: float) -> np.ndarray:
    """Returns the firing rate of neurons.

    Firing rate is defined as ``spike_arr/bin_size``.

    Parameters
    ----------
    spikes_arr: np.ndarray
        A numpy array with spiking data as binary values (1/0).
    bin_size: float
        Time (in seconds).


    Returns
    -------
    np.ndarray
        A numpy array the same size as ``spikes_arr`` but with firing rates.
    """
    return (1 / bin_size) * spikes_arr


def smoothen_firing_rate(
    firing_rate: np.ndarray,
    order: int = 3,
    wn: int = 4000,
    btype: int = "lowpass",
    fs: int = 50000,
) -> np.ndarray:
    """Applies smoothening filter on firing rate.
    """
    b, a = butter(order, wn, btype, fs=fs)

    return filtfilt(b, a, firing_rate)


def plot_firing_rate(
    all_data: np.ndarray,
    session_id: int,
    trial_id: int = None,
    smooth: bool = True,
    by_brain_regions: bool = False,
):
    """Plots the neuron firing rate.

     Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.
    trial_id: int
        Integer that denotes a trial within a session.
    smooth: bool
        Smoothens firing rate if True.
    by_brain_regions: bool
        Produces subplots for each brain-region if True.

    """
    # extract session data.
    session_data = all_data[session_id]

    # create time axis (x-axis).
    dt = session_data["bin_size"]
    T = session_data["spks"].shape[-1]
    time_steps = dt * np.arange(T)

    ###############################################################
    # Subplots with brain regions
    ###############################################################

    if by_brain_regions:

        # ========================================
        # Subplot per brain region for a trial
        # ========================================
        if trial_id:

            (
                idx_region_list,
                all_spikes_arr,
                sorted_neuron_idx,
            ) = sort_neurons_by_brain_region(all_data=all_data, session_id=session_id)

            # important trial events.
            response = session_data["response"][trial_id]
            feedback = session_data["feedback_type"][trial_id]
            contrast_left = session_data["contrast_left"][trial_id]
            contrast_right = session_data["contrast_right"][trial_id]
            decision_type = get_decision_type(response=response, feedback=feedback)

            # for plot info
            idx2response = {-1: "right", 0: "center", 1: "left"}
            idx2feedback = {1: "positive", -1: "negative"}

            # subplots init.
            fig, axes = plt.subplots(
                nrows=len(idx_region_list),
                ncols=1,
                figsize=(15, 15),
                sharey=True,
                squeeze=True,
            )

            start_idx = 0
            for i in range(len(idx_region_list)):
                idx, br = idx_region_list[i]
                end_idx = idx
                spikes_arr = all_spikes_arr[:, trial_id, :][start_idx:end_idx, :]
                start_idx = end_idx

                # base firing rate.
                firing_rate = get_firing_rate(spikes_arr=spikes_arr, bin_size=dt).mean(
                    axis=0
                )

                # smooth the line curves if required.
                if smooth:
                    firing_rate = smoothen_firing_rate(firing_rate)

                axes[i].plot(time_steps, firing_rate)
                axes[i].axvline(
                    x=session_data["stim_onset"], color="red", label="Stimulus Onset"
                )
                axes[i].axvline(
                    x=session_data["gocue"][trial_id], color="green", label="Go Cue"
                )
                axes[i].axvline(
                    x=session_data["response_time"][trial_id],
                    color="orange",
                    label="Response Time",
                )
                axes[i].axvline(
                    x=session_data["feedback_time"][trial_id],
                    color="purple",
                    label="Feedback Time",
                )

                axes[i].set_ylabel("Firing Rate (per 10 msec)", fontsize=12)
                axes[i].set_title(br, fontsize=14)
                axes[i].legend()

            plt.suptitle(
                f"Average Neuron Firing Rate for Session={session_id}  Trial={trial_id}\nDecision Type={decision_type}\nresponse = {idx2response[response]}, feedback={idx2feedback[feedback]}, contrast-left={contrast_left}, contrast-right={contrast_right}",
                fontsize=15,
                va="center",
            )
            plt.xlabel("Time (seconds)", fontsize=12)
            plt.tight_layout()
            plt.show()

        # ========================================
        # Subplot per brain region for a session
        # ========================================

        else:
            (
                idx_region_list,
                all_spikes_arr,
                sorted_neuron_idx,
            ) = sort_neurons_by_brain_region(all_data=all_data, session_id=session_id)

            # important trial events.
            response = session_data["response"]

            # plot figure
            fig, axes = plt.subplots(
                nrows=len(idx_region_list),
                ncols=1,
                figsize=(15, 15),
                sharey=True,
                squeeze=True,
            )

            start_idx = 0
            for i in range(len(idx_region_list)):
                idx, br = idx_region_list[i]
                end_idx = idx
                spikes_arr = all_spikes_arr[start_idx:end_idx, :, :]
                start_idx = end_idx

                # base firing rate.
                firing_rate = get_firing_rate(spikes_arr=spikes_arr, bin_size=dt)

                # event firing rates.
                firing_rate_left = firing_rate[:, response == 1].mean(axis=(0, 1))
                firing_rate_center = firing_rate[:, response == 0].mean(axis=(0, 1))
                firing_rate_right = firing_rate[:, response == -1].mean(axis=(0, 1))

                # smooth the line curves if required.
                if smooth:
                    firing_rate_left = smoothen_firing_rate(firing_rate_left)
                    firing_rate_center = smoothen_firing_rate(firing_rate_center)
                    firing_rate_right = smoothen_firing_rate(firing_rate_right)

                axes[i].plot(time_steps, firing_rate_left, label="Left Turn")
                axes[i].plot(time_steps, firing_rate_center, label="Center Stay")
                axes[i].plot(time_steps, firing_rate_right, label="Right Turn")
                axes[i].axvline(
                    x=session_data["stim_onset"], color="red", label="Stimulus Onset"
                )
                axes[i].set_ylabel("Firing Rate (per 10 msec)", fontsize=12)
                axes[i].set_title(br, fontsize=14)
                axes[i].legend()

            plt.suptitle(
                f"Average Neuron Firing Rate For Session={session_id}",
                fontsize=15,
                va="center",
            )
            plt.xlabel("Time (seconds)", fontsize=12)
            plt.tight_layout()
            plt.show()

    ###############################################################
    # Single plot with all brain regions combined
    ###############################################################

    else:

        # ========================================
        # Subplot per brain region for a trial
        # ========================================
        if trial_id:
            spikes_arr = session_data["spks"][:, trial_id, :]

            # important trial events.
            response = session_data["response"][trial_id]
            feedback = session_data["feedback_type"][trial_id]
            contrast_left = session_data["contrast_left"][trial_id]
            contrast_right = session_data["contrast_right"][trial_id]
            decision_type = get_decision_type(response=response, feedback=feedback)

            # base firing rate.
            firing_rate = get_firing_rate(spikes_arr=spikes_arr, bin_size=dt).mean(
                axis=0
            )

            # smooth the line curves if required.
            if smooth:
                firing_rate = smoothen_firing_rate(firing_rate)

            # for plot info
            idx2response = {-1: "right", 0: "center", 1: "left"}
            idx2feedback = {1: "positive", -1: "negative"}

            # plot figure
            plt.figure(figsize=(15, 5))

            plt.plot(time_steps, firing_rate)

            plt.axvline(
                x=session_data["stim_onset"], color="red", label="Stimulus Onset"
            )
            plt.axvline(
                x=session_data["gocue"][trial_id], color="green", label="Go Cue"
            )
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

            plt.suptitle(
                f"Average Neuron Firing Rate for Session={session_id}  Trial={trial_id}  Decision Type={decision_type}"
            )
            plt.title(
                f"response = {idx2response[response]}, feedback={idx2feedback[feedback]}, contrast-left={contrast_left}, contrast-right={contrast_right}"
            )
            plt.xlabel("Time (seconds)")
            plt.ylabel("Firing Rate (per 10 msec)")

            plt.legend()
            plt.show()

        # ========================================
        # Subplot per brain region for a session
        # ========================================

        else:
            spikes_arr = session_data["spks"]

            # important trial events.
            response = session_data["response"]

            # base firing rate.
            firing_rate = get_firing_rate(spikes_arr=spikes_arr, bin_size=dt)

            # event firing rates.
            firing_rate_left = firing_rate[:, response == 1].mean(axis=(0, 1))
            firing_rate_center = firing_rate[:, response == 0].mean(axis=(0, 1))
            firing_rate_right = firing_rate[:, response == -1].mean(axis=(0, 1))

            # smooth the line curves if required.
            if smooth:
                firing_rate_left = smoothen_firing_rate(firing_rate_left)
                firing_rate_center = smoothen_firing_rate(firing_rate_center)
                firing_rate_right = smoothen_firing_rate(firing_rate_right)

            # plot figure
            plt.figure(figsize=(15, 5))

            plt.plot(time_steps, firing_rate_left, label="Left Turn")
            plt.plot(time_steps, firing_rate_center, label="Center Stay")
            plt.plot(time_steps, firing_rate_right, label="Right Turn")
            plt.axvline(
                x=session_data["stim_onset"], color="red", label="Stimulus Onset"
            )

            plt.title(f"Average Neuron Firing Rate For Session={session_id}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Firing Rate (per 10 msec)")

            plt.legend()
            plt.show()


def reduce_dimensionality(spikes_arr: np.ndarray, n_components: int, algorithm: str):
    """Returns array after peforming dimensionality reduction.

    Parameters
    ----------
    spikes_arr: np.ndarray
        A 2-d numpy array which contains binary spike-data.
    n_components: int
        Final number of components after dimensionality reduction.
    algorithm: str
        Dimensionality reduction algorithm. Available options are - 'pca',
        'tsne', 'factor-analysis', 'fast-ica'.

    Returns
    -------
    np.ndarray
        Numpy array with reduced dimensions.
    """
    if algorithm == "pca":
        spikes_arr_mean = spikes_arr.mean(axis=0)
        pca = PCA(n_components=n_components, random_state=2021).fit(
            (spikes_arr - spikes_arr_mean).T
        )

        spikes_arr_reduced_dim = pca.transform((spikes_arr - spikes_arr_mean).T)
        print(f"Explained variance is = {pca.explained_variance_ratio_}")

    elif algorithm == "tsne":
        spikes_arr_reduced_dim = TSNE(
            n_components=n_components, random_state=2021
        ).fit_transform(spikes_arr.T)

    elif algorithm == "factor-analysis":
        spikes_arr_reduced_dim = FactorAnalysis(
            n_components=n_components, random_state=2021
        ).fit_transform(spikes_arr.T)
    elif algorithm == "fast-ica":
        spikes_arr_reduced_dim = FastICA(
            n_components=n_components, random_state=2021
        ).fit_transform(spikes_arr.T)
    else:
        raise ValueError("Incorrect algorithm type.")

    return spikes_arr_reduced_dim


def plot_components(spikes_arr: np.ndarray, viz_classes: dict):
    """Plot components after reducing dimensions.

    Parameters
    ----------
    spikes_arr: np.ndarray
        A 2-d numpy array which contains binary spike-data.
    viz_classes: dict
        Dicitionary of lists used as an argument to the hue
        parameter of seaborn.
    """
    plt.figure(figsize=(15, 7))
    fig, axes = plt.subplots(nrows=len(viz_classes), ncols=1, figsize=(10, 10))
    for i, k in enumerate(viz_classes):
        sns.scatterplot(
            x=spikes_arr[:, 0], y=spikes_arr[:, 1], hue=viz_classes[k], ax=axes[i]
        )
        axes[i].set_title(f"Reduced neuron dimension coloured by = {k}")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
