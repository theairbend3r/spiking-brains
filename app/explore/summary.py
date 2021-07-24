"""
Functions to summarise session and trial data.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_style("darkgrid")


def trial_event_flow(all_data: np.ndarray, session_id: int, trial_id: int):
    """Prints events in a trial sequentially.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.
    trial_id: int
        Integer that denotes a trial within a session.
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


def session_stats(all_data: np.ndarray, session_id: int):
    """ Prints introductory information about a session and size
    of all variables.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.
    """
    print(f"Number of sessions = {len(all_data)}\n\n")

    print(f"Stats for a session #{session_id}: \n")
    print(
        f"\tNumber of neurons used in this session = {all_data[session_id]['spks'].shape[0]}"
    )
    print(
        f"\tNumber of trials in this session = {all_data[session_id]['spks'].shape[1]}"
    )
    print(f"\tTime taken per trial = {all_data[session_id]['spks'].shape[2]}\n")

    print("-" * 50)
    print("\nData shapes:\n")
    session_keys = all_data[session_id].keys()

    for k in session_keys:
        if type(all_data[session_id][k]) == np.ndarray:
            print(f"\t{k} : {all_data[session_id][k].shape}")
        elif type(all_data[session_id][k]) == list:
            print(f"\t{k} : {len(all_data[session_id][k])}")
        elif type(all_data[session_id][k]) == float:
            print(f"\t{k} :  {all_data[session_id][k]}")


def session_accuracy_report(all_data: np.ndarray, session_id: int, plot: bool) -> float:
    """Returns response accuracy of a mouse in a single session.

    Can optionally plot the confusion matrix.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.
    plot: bool, optional
        Plots a confusion matrix if True.

    Returns
    -------
    float
        Returns response accuracy.
    """
    # -1 for right, +1 for left, 0 for center
    # in session_data["response"]. We remap it to
    # 2, 1, and 0.
    session_data = all_data[session_id]

    idx2class = {2: "right", 0: "center", 1: "left"}

    # 0:center, 1: left, 2:right
    true_output = []
    for l, r in zip(
        session_data["contrast_left"].tolist(), session_data["contrast_right"].tolist()
    ):
        if r > l:
            true_output.append(2)
        elif l > r:
            true_output.append(1)
        else:
            true_output.append(0)

    # 0:center, 1: left, 2:right
    pred_output = session_data["response"].tolist()
    pred_output = [int(i) for i in pred_output]
    pred_output = [2 if i == -1 else i for i in pred_output]

    if plot:
        print(classification_report(true_output, pred_output))
        df = pd.DataFrame(confusion_matrix(true_output, pred_output)).rename(
            columns=idx2class, index=idx2class
        )
        sns.heatmap(df, annot=True)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

    acc = accuracy_score(true_output, pred_output)
    return acc * 100


def session_accuracy(all_data: np.ndarray, session_id: int):
    """Returns the average response accuracy for all trials in a session.

    Uses 'feedback_type' to calculate the accuracy.

    Parameters
    -----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.

    Returns
    -------
    float
        Accuracy percentage.
    """
    session_data = all_data[session_id]
    session_feedback = session_data["feedback_type"]
    session_feedback = np.where(session_feedback == -1, 0, 1)
    session_acc = session_feedback.mean()

    return session_acc * 100


def get_mouse_sessions(all_data: np.ndarray, mouse_name: str) -> list:
    """
    Return session-ids that a single mouse participated in.

    Parameters
    -----------
    all_data: np.ndarray
        3-d numpy array that contains data from all sessions.
    mosue_name: str
        Name of mouse.

    Returns
    -------
    list
        List of sessions for a particular mouse.
    """
    return [i for i in range(len(all_data)) if all_data[i]["mouse_name"] == mouse_name]

