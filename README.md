# Entropy-Constrained-NNK-Means

Repository for the Entropy-Constrained-NNK-Means Algorithm, a modified version of the NNK-Means algorithm introduced in [this paper](https://arxiv.org/abs/2110.08212). This README will guide you through the contents of the repository and how to use the code.

## Table of Contents

- [Introduction](#introduction)
- [Files](#files)
- [Example](#example)

## Introduction

The Entropy-Constrained-NNK-Means Algorithm is a dictionary learning method designed to reduce reliance on hyperparameters and enable a more representive clustering. This repository contains the source code for the algorithm, split across two main files, and a `demo.py` file that demonstrates its usage on the `agnews` dataset.

## Files

The repository is organized as follows:

1. `src/nnk_utils/nnk_means.py`: This file contains the core logic of the Entropy-Constrained-NNK-Means Algorithm

2. `src/NNKMU.py`: This file contains a wrapper class that allows for the algorithm to be used in a similar manner to other scikit-learn clustering algorithms.

3. `src/demo.py`: This file provides an example of how to use the Entropy-Constrained-NNK-Means Algorithm. It loads a sample dataset, applies the clustering algorithm, and stores the resulting cluster assignments.

4. `bin/`: This directory contains a sample dataset that can be used to test the algorithm. The `demo.py` script uses the `agnews` dataset.

## Example

To see the algorithm in action, you can run the provided `demo.py` script. Note that, currently, a CUDA device is required to run the algorithm. Make sure you have the necessary dependencies installed, then execute the script:

```bash
python3 demo.py
```

The script will load a sample dataset, apply the Entropy-Constrained-NNK-Means Algorithm, and save the clustering to the `bin/agnews` folder.

