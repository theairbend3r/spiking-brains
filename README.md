# Spiking Brains

[![Documentation Status](https://readthedocs.org/projects/spiking-brains/badge/?version=latest)](https://spiking-brains.readthedocs.io/en/latest/?badge=latest)

Analysis of the Steinmetz neurons dataset using methods from computational neuroscience and machine learning.

![Spiking Brains](./assets/spiking-brains.png)

## Installation

### Clone the repository.

```sh
git clone https://github.com/theairbend3r/spiking-brains.git
```

### Install the packages.

Using Conda.

```sh
conda env create -f spiking-brains.yml
```

Using Pip.

```sh
pip install requirements.txt
```

## Content

The modules reside in the package `./app`.

Following are the notebooks that use function from `./app/` to perform analysis.

1. `Exploratory Analysis`

   - The `01_exploratory_analysis.ipynb` notebook contains stats about the the dataset.
   - Experiment flow across sessions and trials.
   - Visualization of readings from different brain-regions.

2. `Behaviour Analysis`

   - The `02_behaviour_analysis.ipynb` notebooks contains analysis around response to stimulus.
   - Filter trials by response + feedback and visualize cumulative response accuracy.

3. `Neurons Analysis`

   - The `03_neurons_analysis.ipynb` notebooks contains analysis around neuron spikes and firing rate.
   - Filter neurons by brain regions and visualize spiking activity as raster plots.
   - Dimensionality reduction for visualization of response type and accuracy.

4. `Phenomena Analysis`

   - The `04_phenomenon_analysis.ipynb` notebook analyses spiking activity with respect to behavioural response.
   - Raster plot for devision type vs spiking activity sorted by brain regions.

5. `Machine Learning Modelling`
   - The `05_modelling.ipynb` notebook uses machine learning to study the effect of trial history on current response.

## Experiment and Analysis

### Goal

- Study the effect of memory on current response to behavioral tasks in mice using Python.

### Hypothesis

- Previous responses to visual stimulus, by the mouse, may affect its present response.

### Dataset

- A subset of the Steinmetz dataset (Steinmetz et al, 2019).
- It contains 39 sessions from 10 mice.
- The mice were shown 2 images and had to determine which image had the highest contrast.

### Method

- Train a logistic regression model to predict the mouse's response given the following input variables for `current timestamp - 1`.
  - Feedback type
  - Feedback time
  - Reward time
  - Response type
  - Contrast left
  - Contrast right
- Tune the model and use 8-fold cross validation gauge accuracy.
- Plot the confusion matrix to compare the actual mouse response vs the model's response.
- Analyse the beta-weights per input variable to see it's effect on the response.

### Results

- See `05_modelling.ipynb` notebook.

## Meta

Akshaj Verma – [@theairbend3r](https://twitter.com/theairbend3r).

Distributed under the GNU GPL-V3 license. See `LICENSE` for more information.

[https://github.com/theairbend3r/spiking-brains](https://github.com/theairbend3r/spiking-brains)
