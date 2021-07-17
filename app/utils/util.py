import numpy as np


def filter_neurons_by_brain_area(
    spikes_arr: np.ndarray, brain_areas: list, brain_area_to_neuron: np.ndarray,
) -> np.ndarray:
    """
    Filter neurons by brain areas.

    Input:
        - spikes_arr: 3-d numpy array that contain spiking
            data from a single session. eg. session_data["spks"]
        - brain_areas: a list of brain areas required.
        - brain_area_to_neuron: 1-d numpy array of the same length as
            as the spikes arr at dim=0 which contains a brain area
            string for each index of the array. eg. session_data["brain_area"]

    Output:
        - numpy array with boolen indice that correspond to the presence of a
            neuron in the said area.
    """
    filtered_idx = []

    for ba in brain_areas:
        # get idx of all neurons associated with a single brain area.
        ba_idx = brain_area_to_neuron == ba
        filtered_idx.append(ba_idx)

    filtered_idx = np.array([bool(i) for i in sum(filtered_idx)])

    return filtered_idx


def filter_neurons_by_brain_region(
    spikes_arr: np.ndarray, brain_regions: list, brain_area_to_neuron: np.ndarray
) -> np.ndarray:
    """
    Filter neurons by brain regions.

    Input:
        - spikes_arr: 3-d numpy array that contain spiking
            data from a single session. eg. session_data["spks"]
        - brain_regions: a list of brain regions required.
        - brain_area_to_neuron: 1-d numpy array of the same length as
            as the spikes arr at dim=0 which contains a brain area
            string for each index of the array. eg. session_data["brain_area"]

    Output:
        - numpy array with boolen indice that correspond to the presence of a
            neuron in the said area.
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
            spikes_arr=spikes_arr,
            brain_area_to_neuron=brain_area_to_neuron,
            brain_areas=all_brain_regions[br],
        )

        final_idx.append(filtered_idx)

    return filtered_idx


def sort_neurons_by_brain_region(all_data: np.ndarray, session_id: int):
    """
    Sort the neurons by index based on the brain regions they belong to.

    Input:
        - all_data: 3-d numpy array that contains data for all sessions.
        - session_id: an integer index for the values for a session.

    Output:
        - a tuple of 3 values: idx of last neuron in a region, name of region,
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
            spikes_arr=all_data[session_id]["spks"],
            brain_regions=[br],
            brain_area_to_neuron=all_data[session_id]["brain_area"],
        )

        # get the corresponding ids for all the 'True' items in the
        # bool idx array. These numbers are the neuron ids.
        neuron_id = np.argwhere(bool_idx).squeeze()

        # append both the spike arr associated with the bool_idx and
        # neuron_ids to the list
        sorted_neuron_list.append(neuron_id)
        sorted_spike_list.append(all_data[session_id]["spks"][bool_idx])

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


def filter_trials(session_data: np.ndarray, filter_by: str) -> dict:
    """
    Returns dictionary after filtering the data.

        Input:
            session_data: 3d numpy array with data for a session.
            filter_by: condition for filter. Possible values are
                       ('response', 'feedback', 'response_feedback').

        Output:
            dictionary which contains filtered data.
    """
    if filter_by == "feedback":
        correct = session_data["spks"][:, session_data["feedback_type"] == 1, :]
        wrong = session_data["spks"][:, session_data["feedback_type"] == -1, :]

        return {"correct": correct, "wrong": wrong}

    elif filter_by == "response":
        left = session_data["spks"][:, session_data["response"] == 1, :]
        middle = session_data["spks"][:, session_data["response"] == 0, :]
        right = session_data["spks"][:, session_data["response"] == -1, :]

        return {"left": left, "middle": middle, "right": right}

    elif filter_by == "response_feedback":
        left_idx = session_data["response"] == 1
        middle_idx = session_data["response"] == 0
        right_idx = session_data["response"] == -1

        correct_idx = session_data["feedback_type"] == 1
        wrong_idx = session_data["feedback_type"] == -1

        return {
            "left_correct": session_data["spks"][:, left_idx * correct_idx, :],
            "left_wrong": session_data["spks"][:, left_idx * wrong_idx, :],
            "right_correct": session_data["spks"][:, right_idx * correct_idx, :],
            "right_wrong": session_data["spks"][:, right_idx * wrong_idx, :],
            "middle_correct": session_data["spks"][:, middle_idx * correct_idx, :],
            "middle_wrong": session_data["spks"][:, middle_idx * wrong_idx, :],
        }

    else:
        raise ValueError(
            "filter_by can only be one of - ('response', 'feedback', 'response_feedback')."
        )
