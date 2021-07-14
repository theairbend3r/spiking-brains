import numpy as np


def trial_event_flow(all_data: np.ndarray, session_id: int, trial_id: int):
    """
    Prints events in a trial sequentially.
    """

    session_data = all_data[session_id]

    print(f"Session number = {session_id}")
    print(f"Trial number = {trial_id}/{session_data['spks'].shape[1]}")
    print(f"Contrast left: {session_data['contrast_left'][trial_id]}")
    print(f"Contrast Right: {session_data['contrast_right'][trial_id]}")
    print(f"Response: {session_data['response'][trial_id]}")
    print(f"Feedback: {session_data['feedback_type'][trial_id]}\n")

    print(f"\n{'Time':<8} - {'Action':<10}")
    print("-" * 36)
    print(f"{'0':<8} - {'start':<10}")
    print(f"{session_data['stim_onset']:<8} - {'stim_onset (always fixed)':<10}")
    print(f"{round(session_data['gocue'][trial_id].item(), 3):<8} - {'gocue':<10}")
    print(
        f"{round(session_data['response_time'][trial_id].item(), 3):<8} - {'response_time':<10}"
    )
    print(
        f"{round(session_data['feedback_time'][trial_id].item(), 3):<8} - {'feedback_time':<10}"
    )
    print(f"{'NA':<8} - {'end':<10}")
