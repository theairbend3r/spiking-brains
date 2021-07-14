import numpy as np


def filter_neurons_by_brain_area(
    session_data: dict,
    neuron_brain_areas_mapping: np.ndarray,
    brain_areas: list,
    trial_id: int = None,
) -> tuple:
    """
    Filter neurons by brain areas. Specify trial_id if you want
    to use this function for a single trial data. This returns a
    tuple where first element is the filtered data and second element
    is the filtered indices. The second element is used by the function
    filter_neurons_by_brain_region().
    """
    final_idx = []

    for ba in brain_areas:
        ba_idx = neuron_brain_areas_mapping == ba
        final_idx.append(ba_idx)

    final_idx = [bool(i) for i in sum(final_idx)]

    if trial_id:
        filtered_data = session_data["spks"][final_idx, trial_id, :]
    else:
        filtered_data = session_data["spks"][final_idx, :, :]

    return filtered_data, final_idx


def filter_neurons_by_brain_region(
    session_data: dict, brain_regions: list, trial_id=None
) -> np.ndarray:
    """
    Filter neurons by brain regions. Specify trial_id if you want
    to use this function for a single trial data.
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
        "basal_ganglia ": [
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
    }

    final_idx = []

    for br in brain_regions:
        _, filtered_idx = filter_neurons_by_brain_area(
            session_data=session_data,
            neuron_brain_areas_mapping=session_data["brain_area"],
            brain_areas=all_brain_regions[br],
            trial_id=trial_id,
        )

        final_idx.append(filtered_idx)

    if trial_id:
        filtered_data = session_data["spks"][final_idx[0], trial_id, :]
    else:
        filtered_data = session_data["spks"][final_idx[0], :, :]

    return filtered_data
