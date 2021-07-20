import numpy as np


def session_accuracy(all_data: np.ndarray, session_id: int):
    """
    Returns the average response accuracy for all trials in a session.
    Uses 'feedback_type' to calculate the accuracy.
    """
    session_data = all_data[session_id]
    session_feedback = session_data["feedback_type"]
    session_feedback = np.where(session_feedback == -1, 0, 1)
    session_acc = session_feedback.mean()

    return session_acc * 100


def get_mouse_sessions(all_data: np.ndarray, mouse_name: str):
    """
    Return session-ids that a single mouse participated in.
    """
    return [i for i in range(len(all_data)) if all_data[i]["mouse_name"] == mouse_name]

