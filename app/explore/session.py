import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_style("darkgrid")


def session_stats(all_data: np.ndarray, session_id: int):
    """
    Prints introductory information about a session and size
    of all variables.
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


# def session_accuracy_report(session_data: dict, plot: bool) -> float:
#     """
#     Returns accuracy or plots the decision summary of a session.
#     """
#     # -1 for right, +1 for left, 0 for center
#     # in session_data["response"]. We remap it to
#     # 2, 1, and 0.

#     idx2class = {2: "right", 0: "center", 1: "left"}

#     # dsicard for accuracy calculation where the left and right contrast are
#     # equal and non-zero as the mouse was randomly rewarded for those cases.

#     keep_idx = np.logical_not(
#         (session_data["contrast_left"] == session_data["contrast_right"])
#         * (session_data["contrast_left"] != 0)
#         * (session_data["contrast_right"] != 0)
#     )

#     # 0:center, 1: left, 2:right
#     true_output = []
#     for l, r in zip(
#         session_data["contrast_left"].tolist(), session_data["contrast_right"].tolist()
#     ):
#         if r > l:
#             true_output.append(2)
#         elif l > r:
#             true_output.append(1)
#         else:
#             true_output.append(0)

#     # 0:center, 1: left, 2:right
#     pred_output = session_data["response"].tolist()
#     pred_output = [int(i) for i in pred_output]
#     pred_output = [2 if i == -1 else i for i in pred_output]

#     # ignore the equal but non-zero contrast data.
#     pred_output = np.array(true_output)[keep_idx]
#     true_output = np.array(true_output)[keep_idx]

#     if plot:
#         print(classification_report(true_output, pred_output))
#         df = pd.DataFrame(confusion_matrix(true_output, pred_output)).rename(
#             columns=idx2class, index=idx2class
#         )
#         sns.heatmap(df, annot=True)
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.title("Confusion Matrix")
#     else:
#         acc = accuracy_score(true_output, pred_output)
#         return acc * 100


def session_accuracy_report(session_data: dict, plot: bool) -> float:
    """
    Returns accuracy or plots the decision summary of a session.
    """
    # -1 for right, +1 for left, 0 for center
    # in session_data["response"]. We remap it to
    # 2, 1, and 0.

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
    else:
        acc = accuracy_score(true_output, pred_output)
        return acc * 100

