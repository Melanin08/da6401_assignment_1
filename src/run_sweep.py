import os
import random

datasets = ["mnist"]
optimizers = ["sgd", "momentum", "nag", "rmsprop"]
activations = ["relu", "sigmoid", "tanh"]
learning_rates = [0.01, 0.001, 0.0005]
batch_sizes = [32, 64, 128]

hidden_configs = [
    "64 64",
    "128 128",
    "128 128 128"
]

weight_inits = ["random", "xavier"]  

runs = 100

for i in range(runs):

    dataset = random.choice(datasets)
    optimizer = random.choice(optimizers)
    activation = random.choice(activations)
    lr = random.choice(learning_rates)
    batch = random.choice(batch_sizes)
    hidden = random.choice(hidden_configs)
    init = random.choice(weight_inits)   

    cmd = f"python train.py -d {dataset} -e 5 -b {batch} -lr {lr} -o {optimizer} -a {activation} -sz {hidden} -l cross_entropy -w_i {init}"

    print("Running:", cmd)
    os.system(cmd)