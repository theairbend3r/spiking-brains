"""
Functions to work with logistic regression.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from app.utils.util import sort_neurons_by_brain_region, map_neuron_idx_to_region
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_model(x: np.ndarray, y: np.ndarray, plot: bool = False):
    """Returns a trained model.

    Parameters
    ----------
    x: np.ndarray
        2-d numpy array of shape (n_samples, n_features) corresponding to inputs.
    y: np.ndarray
        1-d numpy array of shape (n_samples, ) corresponding to outputs.
    plot: bool
        Plots accuracy box plot of k-fold crossvalidation if True.


    Returns
    -------
    tuple
        A tuple of trained model and output predictions.

    """
    clf = LogisticRegression(
        random_state=2021, penalty="l1", solver="liblinear", max_iter=1000
    ).fit(x, y)
    y_pred = clf.predict(x)

    accuracies = cross_val_score(clf, x, y, cv=8)

    print(
        f"Median accuracy in k-fold cross validation = {round(np.median(accuracies) * 100, 3)}%"
    )

    if plot:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=accuracies * 100)
        plt.xlabel("Accuracy")
        plt.title("8-Fold Cross Validation Accuracies")

    return clf, y_pred


def map_neuron_weights_to_brain_region(
    all_data: np.ndarray, session_id: int, model
) -> dict:
    """
    Returns a dictionary with (neuron-idx, brain-region) tuples associated
    with each response type.

    Parameters
    ----------
    all_data: np.ndarray
        A 2-d numpy array that contains data from all sessions.
    session_id: int
        Integer that denotes a particular session.

    Returns
    -------
    dict
        Corresponding neuron and brain-region for each response type.
    """

    # convert the class id to strings
    idx2response = {i: k for i, k in enumerate(model.classes_)}

    # sort idx by brain region and get the last
    # index of each neuron in a ragion
    brain_region_last_idx = sort_neurons_by_brain_region(
        all_data=all_data, session_id=session_id
    )[0]

    # create a list of indices for neurons with non-zero weights
    # associated with each response type (dict)
    response_to_neuron_idx_dict = {}

    for i in range(len(model.classes_)):
        non_zero_response_neuron_idx = np.where(model.coef_[i, :] != 0)
        response_to_neuron_idx_dict[idx2response[i]] = non_zero_response_neuron_idx[
            0
        ].tolist()

    # Classify the neuron indices into brain regions by using the
    # brain_region_last_idx.
    response_to_region_dict = {k: [] for k in model.classes_}

    for res in model.classes_:
        for idx in response_to_neuron_idx_dict[res]:
            region = map_neuron_idx_to_region(
                neuron_idx=idx, brain_region_last_idx=brain_region_last_idx
            )
            response_to_region_dict[res].append((idx, region))

    return response_to_region_dict


def plot_confusion_matrix(y_true: list, y_pred: list):
    """Plots the confusion matrix.

    Parameters
    ----------
    y_true: list
        A 1-d numpy array or list of true values.
    y_pred: list
        A 1-d numpy array or list of predicted values.
    """

    idx2class = {2: "right", 0: "left", 1: "middle"}

    df = pd.DataFrame(confusion_matrix(y_true, y_pred)).rename(
        columns=idx2class, index=idx2class
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt="g", cbar=False, annot_kws={"fontsize": 18})
    plt.xlabel("Model's Prediction", fontsize=18)
    plt.ylabel("Actual Mouse Response", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
