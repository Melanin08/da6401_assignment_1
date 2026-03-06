# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

This assignment requires you to implement a neural network from scratch using only NumPy. You will build all components including layers, activations, optimizers, and loss functions, then train your network on MNIST or Fashion-MNIST datasets.

## Learning Objectives

- Understand forward and backward propagation
- Implement gradient computation manually
- Implement various optimizers (SGD, Momentum, NAG, RMSprop)
- Work with activation functions and their derivatives
- Train and evaluate neural networks
- Log experiments using Weights & Biases

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py -d mnist -e 10 -b 64 -lr 0.01 -o sgd -nhl 2 -sz 128 128 -a relu -l cross_entropy -wi xavier
```

**Arguments:**
- `-d, --dataset`: Dataset to use (mnist/fashion_mnist)
- `-e, --epochs`: Number of training epochs
- `-b, --batch_size`: Mini-batch size
- `-lr, --learning_rate`: Initial learning rate
- `-wd, --weight_decay`: Weight decay for L2 regularization
- `-o, --optimizer`: Optimizer (sgd/momentum/nag/rmsprop)
- `-nhl, --num_layers`: Number of hidden layers
- `-sz, --hidden_size`: Hidden layer sizes (space-separated)
- `-a, --activation`: Activation function (relu/sigmoid/tanh)
- `-l, --loss`: Loss function (cross_entropy/mse)
- `-wi, --weight_init`: Weight initialization (random/xavier)

### Inference

```bash
python src/inference.py -model_path best_model.npy -config_path best_model_config.json -d mnist
```

### Data Exploration

```bash
python src/data_exploration.py
```

### Hyperparameter Sweep

```bash
python src/run_sweep.py
```

## Project Structure

```
├── src/
│   ├── ann/                    # Neural network components
│   │   ├── __init__.py
│   │   ├── activations.py      # Activation functions
│   │   ├── neural_layer.py     # Dense layer implementation
│   │   ├── neural_network.py   # Main network class
│   │   ├── objective_functions.py  # Loss functions
│   │   └── optimizers.py       # Optimization algorithms
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Data loading utilities
│   ├── data_exploration.py     # W&B data visualization
│   ├── inference.py           # Model evaluation
│   ├── run_sweep.py          # Hyperparameter sweep
│   └── train.py              # Training script
├── models/                   # Saved models
├── notebooks/               # Jupyter notebooks
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Requirements

- NumPy
- Keras (for data loading only)
- Matplotlib
- Scikit-learn
- Weights & Biases

## Contact

For questions or issues, please contact the teaching staff or post on the course forum.

---

Good luck with your implementation!
