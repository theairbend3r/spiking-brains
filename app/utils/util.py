"""
Functions for general utilities.
"""

import numpy as np


def get_decision_type(response: int, feedback: int) -> str:
    """Returns decision type string.

    Parameters
    ----------
    response: int
        Response associated with a single trial. Can be -1 (right),
        0 (middle), (1)left.
    feedback: int
        Feedback associated with the given response for a single trial.
        Can be +1 (correct) or -1 (wrong).

    Returns
    -------
    str
        ``response``_``feeback``
    """
    idx2response = {-1: "right", 0: "center", 1: "left"}
    idx2feedback = {1: "correct", -1: "wrong"}

    response = idx2response[response]
    feedback = idx2feedback[feedback]

    return response + "_" + feedback


def filter_neurons_by_brain_area(
    brain_areas: list, brain_area_to_neuron: np.ndarray,
) -> np.ndarray:
    """Filter neurons by brain areas.

    Parameters
    ----------
    brain_areas: list
        A list of brain areas required.
    brain_area_to_neuron: np.ndarray
        1-d numpy array of the same length as as the spikes arr at
        dim=0 which contains a brain area string for each index of
        the array. eg. session_data["brain_area"].

    Returns
    -------
    np.ndaray
        A numpy array with boolen indice that correspond to the
        presence of a neuron in the said area.
    """
    filtered_idx = []

    for ba in brain_areas:
        # get idx of all neurons associated with a single brain area.
        ba_idx = brain_area_to_neuron == ba
        filtered_idx.append(ba_idx)

    filtered_idx = np.array([bool(i) for i in sum(filtered_idx)])

    return filtered_idx


def filter_neurons_by_brain_region(
    brain_regions: list, brain_area_to_neuron: np.ndarray
) -> np.ndarray:
    """Filter neurons by brain regions.

    Parameters
    ----------
    brain_regions: list
        A list of brain regions required.
    brain_area_to_neuron: np.ndarray
        1-d numpy array of the same length as as the spikes arr at
        dim=0 which contains a brain area string for each index of
        the array. eg. session_data["brain_area"].

    Returns
    -------
    np.ndaray
        A numpy array with boolen indice that correspond to the
        presence of a neuron in the said region.
    """

    all_brain_regions = {
        "visual_cortex": ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
        "thalamus": [
            "CL",
            "LD",
            "LGd",
            "LH",
            "LP",
            "MD",
            "MG",
            "PO",
            "POL",
            "PT",
            "RT",
            "SPF",
            "TH",
            "VAL",
            "VPL",
            "VPM",
        ],
        "hippocampas": ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],
        "non_visual_cortex": [
            "ACA",
            "AUD",
            "COA",
            "DP",
            "ILA",
            "MOp",
            "MOs",
            "OLF",
            "ORB",
            "ORBm",
            "PIR",
            "PL",
            "SSp",
            "SSs",
            "RSP",
            "TT",
        ],
        "midbrain": [
            "APN",
            "IC",
            "MB",
            "MRN",
            "NB",
            "PAG",
            "RN",
            "SCs",
            "SCm",
            "SCig",
            "SCsg",
            "ZI",
        ],
        "basal_ganglia": [
            "ACB",
            "CP",
            "GPe",
            "LS",
            "LSc",
            "LSr",
            "MS",
            "OT",
            "SNr",
            "SI",
        ],
        "cortical_subplate": ["BLA", "BMA", "EP", "EPd", "MEA"],
        "other": ["root"],
    }

    final_idx = []

    for br in brain_regions:
        # get idx of all neurons associated with all areas in a
        # single brain region.
        filtered_idx = filter_neurons_by_brain_area(
            brain_area_to_neuron=brain_area_to_neuron,
            brain_areas=all_brain_regions[br],
        )

        final_idx.append(filtered_idx)

    return filtered_idx


def sort_neurons_by_brain_region(all_data: np.ndarray, session_id: int):
    """Sort the neurons by index based on the brain regions they belong to.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.

    Returns
    -------
    tuple
        a tuple of 3 values: idx of last neuron in a region, name of region,
        and a sorted index array of neurons clubbed by all brain regions.
    """

    # contains the actual spike data
    sorted_spike_list = []

    # contains the neuron ids aka the the ids of the sorted spikes data
    sorted_neuron_list = []

    all_brain_regions = [
        "visual_cortex",
        "thalamus",
        "hippocampas",
        "non_visual_cortex",
        "midbrain",
        "basal_ganglia",
        "cortical_subplate",
        "other",
    ]

    for br in all_brain_regions:
        # get boolean index per brain region
        bool_idx = filter_neurons_by_brain_region(
            brain_regions=[br], brain_area_to_neuron=all_data[session_id]["brain_area"],
        )

        # get the corresponding ids for all the 'True' items in the
        # bool idx array. These numbers are the neuron ids.
        neuron_id = np.argwhere(bool_idx).squeeze()

        # append both the spike arr associated with the bool_idx and
        # neuron_ids to the list
        sorted_neuron_list.append(neuron_id)

        # sorted neurons in the brain region by the number of spikes
        region_specific_spikes_arr = all_data[session_id]["spks"][bool_idx]
        sorted_region_specific_spikes_arr = region_specific_spikes_arr[
            np.argsort(region_specific_spikes_arr.sum(axis=(1, 2)))[::-1]
        ]
        sorted_spike_list.append(sorted_region_specific_spikes_arr)

    # flatten out the list of list into a single numpy array of neuron ids
    # sorted by brain region.
    sorted_neuron_idx = np.hstack(sorted_neuron_list)

    # get the len of each spike array and apply cumsum on them to obtain
    # the last index of each region. For example, the length of 3 brain
    # regions are 100, 50, and 70. This means the last neuron has index
    # of 100, 150, and 220. We want this to deliniate the different
    # brain regions wrt to number of neurons.
    arr_len = [spike_arr.shape[0] for spike_arr in sorted_spike_list]
    all_idx = [sum(arr_len[0:i]) for i in range(1, len(arr_len) + 1)]

    # convert spike data list of list into a single array
    sorted_spike_arr = np.vstack(sorted_spike_list)

    # obtain tuple of final index and brain region by removing those regions
    # with 0 values.
    idx_region = [
        (idx, br)
        for arr_len, idx, br in zip(arr_len, all_idx, all_brain_regions)
        if arr_len != 0
    ]

    return idx_region, sorted_spike_arr, sorted_neuron_idx


def filter_trials(all_data: np.ndarray, session_id: int, filter_by: str) -> dict:
    """
    Returns dictionary of indices after filtering the data.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.
    filter_by: str
        condition for filter. Possible values are - 'response', 'feedback',
        or 'response_feedback'.

    Returns
    -------
    dict
        dictionary which contains boolean indices arrays.
    """
    session_data = all_data[session_id]

    if filter_by == "feedback":
        correct = session_data["feedback_type"] == 1
        wrong = session_data["feedback_type"] == -1

        return {"correct": correct, "wrong": wrong}

    elif filter_by == "response":
        left = session_data["response"] == 1
        middle = session_data["response"] == 0
        right = session_data["response"] == -1

        return {"left": left, "middle": middle, "right": right}

    elif filter_by == "response_feedback":
        left_idx = session_data["response"] == 1
        middle_idx = session_data["response"] == 0
        right_idx = session_data["response"] == -1

        correct_idx = session_data["feedback_type"] == 1
        wrong_idx = session_data["feedback_type"] == -1

        return {
            "left_correct": left_idx * correct_idx,
            "left_wrong": left_idx * wrong_idx,
            "right_correct": right_idx * correct_idx,
            "right_wrong": right_idx * wrong_idx,
            "middle_correct": middle_idx * correct_idx,
            "middle_wrong": middle_idx * wrong_idx,
        }

    else:
        raise ValueError(
            "filter_by can only be one of - ('response', 'feedback', 'response_feedback')."
        )


def map_neuron_idx_to_region(neuron_idx: int, brain_region_last_idx: list) -> str:
    """ Returns associated brain region given a neuron index in a single session.

    Parameters
    ----------
    neuron_idx: int
        Index of neuron in the spikes array of a single session.
    brain_region_last_idx: list
        Last neuron index associated with a brain region.

    Returns
    -------
    str
        Brain region of the given neuron index.
    """
    min_idx = 0
    for i in range(0, len(brain_region_last_idx)):
        if neuron_idx >= min_idx and neuron_idx <= brain_region_last_idx[i][0]:
            return brain_region_last_idx[i][1]

        min_idx = brain_region_last_idx[i][0]

    else:
        return f"No corresponding brain region found for neuron #{neuron_idx}."

