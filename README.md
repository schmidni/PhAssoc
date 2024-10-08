# Phase Association

## Description
This repository contains the code for testing and evaluating various phase association algorithms. It also includes plotting utilities for generating the graphs for the report "Machine Learning Phase Association for Underground Laboratory Acoustic Emission Data".

## Models
The repository contains implementations of [GaMMA](https://github.com/AI4EPS/GaMMA) and [PyOcto](https://github.com/yetinam/pyocto), both tested using synthetic data.

A new method employing contrastive learning with a simple neural network using NTXentLoss has also been explored. The aim is to separate clusters in the embedding space and subsequently apply a basic clustering algorithm for cluster identification. While promising, this method was not included in the final report due to the need for further refinement, particularly in handling noise picks and determining cluster counts.

## Scripts

### `create_synthetic_data.py`
Generates synthetic data for testing phase association algorithms. This script is designed for easy configuration.

### `models_gmm.py`
Provides a basic implementation of the scikit learn functions of the standard Bayesian Gaussian Mixture Model for clustering.

### `models_gamma.py`, `models_pyocto.py`
These scripts contain the implementations for the GaMMA and PyOcto models, respectively, utilizing data generated by `create_synthetic_data.py`.

### `models_compare.py`, `models_gamma_test.py`
Scripts for calculating and comparing performance metrics. These scripts generate ad-hoc predefined synthetic data for model testing.

### `models_contrastive_test.py`, `models_contrastive_train.py`
Scripts for the implementation and testing of the contrastive learning method.

### `plots_data.py`, `plots_picks.py`
Scripts for visualizing properties of the synthetic data created for testing.

## Source Code Organization
Source code is organized within the `src` directory as follows:

### `src/gamma`
Contains the modified implementation of GaMMA.

### `src/clustering`
- `dataset.py`: Custom torch.dataset class for loading synthetic data.
- `utils.py:ClusterStatistics`: Class for calculating performance metrics of clustering algorithms.

### `src/synthetics`
Contains scripts for generating synthetic data.


## Disclaimer
ChatGPT as well as Github Copliot was used to support the development of this code.