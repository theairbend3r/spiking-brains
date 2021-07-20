import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def train_model(x:np.ndarray, y:np.ndarray, plot:bool=False):
    """
    Returns a trained model. 
    
    Input:
        - x: input array.
        - y: output array.
        - plot: plots box plot of cross validation accuracies.
        
    Output:
        - trained model.
    """
    clf = LogisticRegression(random_state=2021, penalty="l1", solver="liblinear", max_iter=1000).fit(x, y)
    y_pred = clf.predict(x)

    accuracies = cross_val_score(clf, x, y, cv=8)

    print(f"Median accuracy in k-fold cross validation = {round(np.median(accuracies) * 100, 3)}%")

    if plot:
        sns.boxplot(x = accuracies * 100)
        plt.title("Accuracy Boxplot")


    return clf


def plot_beta_weights(model, input_name_list:list=None):
    """
    Plots a bar-chart with beta weights and associated input features.
    
    Input:
        - model: trained sklearn, logistic regression, model.
        - input_name_list (optional): list of variable names.
    """
    fig, axes = plt.subplots(nrows=len(model.classes_), ncols=1, figsize=(15, 12), sharey=True)
    
    odds = np.exp(model.coef_)
    probs = odds/(1 + odds)
    for i, c in enumerate(model.classes_):
        weights_response = probs[i, :]
        
        if input_name_list:
            sns.barplot(x=input_name_list, 
                        y=weights_response.squeeze(),
                        ax=axes[i])
            axes[i].set_title(f"Response = {c}")
            axes[i].set_ylabel("Beta Weights")
        else:
            input_idx_response = np.argwhere(weights_response != 0.5)
            sns.barplot(x=input_idx_response.squeeze(), 
                        y=weights_response[input_idx_response].squeeze(),
                        ax=axes[i])
        
            axes[i].set_title(f"Response = {c}")
            axes[i].set_ylabel("Beta Weights")
        
    plt.suptitle(f"Beta Weights For Different Response Types")