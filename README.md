# Assignment 1: Multi-Layer Perceptron for Image Classification

Course: DA6401 – Introduction to Deep Learning  
Student: Ayman Hamza Haji 

# Overview

This project implements a **Multi-Layer Perceptron (MLP)** neural network from scratch using **NumPy only**. The objective of this assignment is to understand the internal working of neural networks by manually implementing all major components including:

- forward propagation
- backward propagation
- gradient computation
- activation functions
- loss functions
- optimization algorithms
- weight initialization methods

The model is trained on the **MNIST dataset** and evaluated on the **Fashion-MNIST dataset**. Experiments and results are tracked using **Weights & Biases (W&B)**.


# Learning Objectives

The main learning goals of this assignment are:

- Understand forward propagation in neural networks
- Implement backpropagation and manual gradient computation
- Implement activation functions and their derivatives
- Implement optimization algorithms
- Train neural networks from scratch
- Perform hyperparameter tuning
- Track and analyze experiments using Weights & Biases


# Datasets

Two datasets were used in this project.

## MNIST

The MNIST dataset contains grayscale images of handwritten digits from **0 to 9**.

Image size: **28 × 28**  
Number of classes: **10**

## Fashion-MNIST

Fashion-MNIST contains grayscale images of clothing items such as:

- shirts
- trousers
- sneakers
- bags
- coats

Fashion-MNIST is more complex than MNIST because some classes have very similar visual patterns.

Both datasets are loaded using **keras.datasets**.


# Installation

Install all required libraries using:

pip install -r requirements.txt


# Project Structure
# Project Structure

```
da6401_assignment_1/
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── wandb_demo.ipynb
│
├── src/
│   │
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   │
│   ├── train.py
│   ├── inference.py
│   ├── q21_data_exploration.py
│   ├── q22_hyperparameter_sweep.py
│   ├── q24_vanishing_gradient.py
│   └── q28_error_analysis.py
│
├── best_config.json
├── best_model.npy
│
├── requirements.txt
└── README.md
```
<<<<<<< HEAD

=======
>>>>>>> 0d5519b525ef9163cb4ce05ce3af1722ad85a241
# File Description

## activations.py

Implements activation functions and their derivatives:

- ReLU
- Sigmoid
- Tanh
- Softmax


## neural_layer.py

Implements a dense neural network layer including:

- weight initialization
- forward pass
- backward pass
- gradient storage

## neural_network.py

Defines the complete MLP network and manages:

- network construction
- forward propagation
- backward propagation
- evaluation
- saving and loading models

## objective_functions.py

Implements loss functions:

- Cross Entropy
- Mean Squared Error (MSE)


## optimizers.py

Implements optimization algorithms:

- SGD
- Momentum
- NAG
- RMSProp


## data_loader.py

Loads and preprocesses datasets:

- MNIST
- Fashion-MNIST

Also performs normalization and one-hot encoding.

## train.py

Main training script responsible for:

- parsing command line arguments
- loading dataset
- training loop
- logging metrics to W&B
- validation and testing
- saving best model


## inference.py

Loads a trained model and evaluates it on the dataset.  
Outputs:

- accuracy
- precision
- recall
- F1 score
- loss


# Training

Example command to train the model:

python src/train.py -d mnist -e 10 -b 32 -lr 0.001 -o rmsprop -wd 0.0001 -nhl 2 -sz 128 128 -a relu -l cross_entropy -wi xavier


# Command Line Arguments

-d dataset (mnist or fashion_mnist)

-e number of training epochs

-b batch size

-lr learning rate

-o optimizer (sgd, momentum, nag, rmsprop)

-wd weight decay for regularization

-nhl number of hidden layers

-sz hidden layer sizes

-a activation function (relu, sigmoid, tanh)

-l loss function (cross_entropy or mse)

-wi weight initialization (random, xavier, zeros)


# Inference

To evaluate a saved model run:

python src/inference.py --model_path src/src/best_model.npy -d mnist

This prints:

Accuracy  
Precision  
Recall  
F1 Score  
Loss  


# Experiments Performed

Several experiments were conducted as part of this assignment.


# 2.1 Dataset Visualization

Sample images from the dataset were visualized using W&B to understand class distribution and dataset structure.

# 2.2 Hyperparameter Sweep

A W&B sweep with **more than 100 runs** was performed exploring:

- learning rates
- optimizers
- activation functions
- hidden layer sizes
- weight initialization methods

Parallel coordinate plots were used to identify the most important hyperparameters.

Best configuration found:

Dataset: MNIST  
Epochs: 10  
Batch Size: 32  
Learning Rate: 0.001  
Optimizer: RMSProp  
Hidden Layers: [128,128]  
Activation: ReLU  
Loss Function: Cross Entropy  

Best Test Accuracy: **97.57%**

# 2.3 Optimizer Comparison

The following optimizers were compared:

- SGD
- Momentum
- NAG
- RMSProp

All models used the same architecture:

3 hidden layers  
128 neurons per layer  
ReLU activation

RMSProp converged fastest during early training epochs.

# 2.4 Vanishing Gradient Analysis

Sigmoid and ReLU activation functions were compared using RMSProp.

The gradient norm of the first hidden layer was logged during training.

Observations:

- Sigmoid networks suffered from vanishing gradients
- ReLU maintained stronger gradient values

# 2.5 Dead Neuron Investigation

Using ReLU activation with a high learning rate caused some neurons to output zero for all inputs.

These neurons stopped learning and became **dead neurons**.

Using **Tanh activation** prevented this issue because neurons continued producing non-zero outputs.

# 2.6 Loss Function Comparison

Two loss functions were compared:

- Mean Squared Error
- Cross Entropy Loss

Cross Entropy converged faster and achieved higher accuracy because it is better suited for classification tasks.

# 2.7 Global Performance Analysis

Training accuracy and test accuracy across runs were analyzed.

Some models achieved high training accuracy but lower test accuracy, indicating **overfitting**.


# 2.8 Error Analysis

A confusion matrix was generated for the best performing model.

Most digits were classified correctly.  
Some misclassifications occurred between visually similar digits such as:

- 4 and 9
- 3 and 5

Misclassified images were also visualized.

# 2.9 Weight Initialization and Symmetry Breaking

Two initialization strategies were compared:

- Zero initialization
- Xavier initialization

Zero initialization caused identical gradients for all neurons.

Xavier initialization broke symmetry and allowed neurons to learn different features.


# 2.10 Fashion-MNIST Transfer Challenge

Three configurations based on MNIST experiments were tested on Fashion-MNIST.

Configuration 1

Architecture: [128,128]  
Optimizer: RMSProp  
Activation: ReLU  

Test Accuracy: **86.38%**

Configuration 2

Architecture: [128,128,128]  
Optimizer: RMSProp  
Activation: ReLU  

Test Accuracy: **86.64%**

Configuration 3

Architecture: [128,128]  
Optimizer: NAG  
Activation: Tanh  

Test Accuracy: **85.78%**

The deeper architecture performed slightly better due to the increased complexity of Fashion-MNIST images.


# Results

MNIST Test Accuracy ≈ **97.5%**

Fashion-MNIST Test Accuracy ≈ **86.6%**


# Weights & Biases Report

Full report available at:
<<<<<<< HEAD

=======
>>>>>>> 0d5519b525ef9163cb4ce05ce3af1722ad85a241
https://wandb.ai/ge26z814-iitm-india/da6401-assignment1/reports/Multi-Layer-Perceptron-for-Image-Classification--VmlldzoxNjEzMjU1MQ?accessToken=88b1uzf5miahd7fqh2l61rvbd8erdgzviwygy4vx4hue1sev2d7x6jmnbpf1m87v

# Notes

- Neural network implemented entirely using **NumPy**
- No deep learning frameworks such as TensorFlow or PyTorch were used
- Forward propagation, backward propagation, and optimizer updates were implemented manually
