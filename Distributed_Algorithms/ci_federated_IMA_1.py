import numpy as np
import torch
from torch.autograd import Variable
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import algorithm.optimizer as opt
from sklearn.preprocessing import StandardScaler
import pandas as pd


np.random.seed(0)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def consensus_innovation(weights, datapoints, adj_matrix, learning_rate=0.1, lambda_reg=0):
    num_nodes = adj_matrix.shape[0]
    num_features = weights.shape[1]  # This should be 1

    weights_new = np.copy(weights)

    for i in range(num_nodes):
        consensus_sum = np.zeros(num_features)
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0:
                consensus_sum += adj_matrix[i, j] * weights[j]

        gradient = np.dot(datapoints[i]['features'].T,
                          (np.dot(datapoints[i]['features'], weights[i]) - datapoints[i]['label']))
        regularization_term = lambda_reg * weights[i]
        gradient += regularization_term

        weights_new[i] = consensus_sum - learning_rate * gradient

    return weights_new


def federated_learning_with_consensus_innovation_and_ima(initial_local_models, num_rounds, datapoints, adj_matrix,
                                                         num_clients=200, client_fraction=0.9,
                                                         learning_rate=0.15, alpha=0.1, lambda_reg=0):
    global_model = np.copy(np.mean(initial_local_models, axis=0))  # Initialize global model
    mse_list = []  # To store MSE values for each round
    iteration_scores = []

    num_features = global_model.shape[0]  # This will be 1

    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")

        # Randomly select a fraction of clients
        selected_clients = np.random.choice(range(num_clients),
                                            size=int(client_fraction * num_clients),
                                            replace=False)

        # Placeholder for aggregated model weights
        local_weights = np.copy(initial_local_models)
        updated_weights = np.copy(initial_local_models)

        for client in selected_clients:
            # For each client, update the local weights using the global model
            local_weights[client] = global_model

        # Apply consensus innovation on selected clients
        updated_weights = consensus_innovation(local_weights, datapoints, adj_matrix, learning_rate, lambda_reg)

        # Aggregate weights
        agg_weights = np.mean(updated_weights[selected_clients], axis=0)

        # IMA step: Update global model
        global_model = alpha * global_model + (1 - alpha) * agg_weights

        Y_pred = np.array([datapoints[i]['features'] @ updated_weights[i] for i in range(num_clients)])
        true_labels = np.array([datapoints[i]['label'] for i in range(num_clients)])
        mse = mean_squared_error(true_labels, Y_pred)
        iteration_scores.append(mse)

    print(global_model)
    print('----------------------------------------')
    print(updated_weights)
    print('----------------------------------------')
    print(agg_weights)
    return iteration_scores, global_model


def get_sbm_data(cluster_sizes, G, W, m=1, n=2, noise_sd=0, is_torch_model=True):
    N = len(G.nodes)
    E = len(G.edges)

    B = np.zeros((E, N))
    weight_vec = np.zeros(E)
    cnt = 0
    for i, j in G.edges:
        if i > j:
            continue
        B[cnt, i] = 1
        B[cnt, j] = -1
        weight_vec[cnt] = 1
        cnt += 1

    weight_vec = weight_vec[:cnt]
    B = B[:cnt, :]

    node_degrees = np.array(np.sum(abs(B), 0)).ravel()
    datapoints = {}
    true_labels = []

    cnt = 0
    for i, cluster_size in enumerate(cluster_sizes):
        for j in range(cluster_size):
            features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
            label = np.dot(features, W[i]) + np.random.normal(0, noise_sd)
            true_labels.append(label)

            if is_torch_model:
                model = opt.TorchLinearModel(n)
                optimizer = opt.TorchLinearOptimizer(model)
                features = Variable(torch.from_numpy(features)).to(torch.float32)
                label = Variable(torch.from_numpy(label)).to(torch.float32)
            else:
                model = opt.LinearModel(node_degrees[i], features, label)
                optimizer = opt.LinearOptimizer(model)

            datapoints[cnt] = {
                'features': features,
                'degree': node_degrees[i],
                'label': label,
                'optimizer': optimizer
            }
            cnt += 1

    return B, weight_vec, np.array(true_labels), datapoints


def incidence_to_adjacency(incidence_matrix):
    num_edges, num_nodes = incidence_matrix.shape
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge_idx in range(num_edges):
        node_indices = np.where(incidence_matrix[edge_idx, :] != 0)[0]
        if len(node_indices) == 2:
            adjacency_matrix[node_indices[0], node_indices[1]] = 1
            adjacency_matrix[node_indices[1], node_indices[0]] = 1

    return adjacency_matrix


def calculate_a_kl(k, l, matrix, degrees):
    if k != l:
        if matrix[k][l] == 1:
            return 1 / max(degrees[k], degrees[l])
        else:
            return 0
    else:
        return 1 - sum(
            calculate_a_kl(k, i, matrix, degrees) for i in range(len(matrix)) if i != k and matrix[k][i] == 1)


def create_a_matrix(matrix, degrees):
    size = len(matrix)
    a_matrix = np.zeros((size, size))
    for k in range(size):
        for l in range(size):
            a_matrix[k][l] = calculate_a_kl(k, l, matrix, degrees)
    return a_matrix


def get_sbm_2blocks_data(weight, m=1, n=2, pin=0.5, pout=0.01, noise_sd=0, is_torch_model=True):
    cluster_sizes = [100, 100]
    probs = np.array([[pin, pout], [pout, pin]])
    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)

    W1 = np.array([weight, 2])
    W2 = np.array([-weight, 2])
    W = [W1, W2]

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
    num_clients = 200
    num_features = 2  # Adjust this to match your specific model's requirements
    temp = np.zeros(num_features)  # Initialize with correct dimensions for 1D weights
    initial_local_models = np.stack([temp] * num_clients, axis=0)  # Create an array of shape (num_clients, num_features)
    total_error, _ = federated_learning_with_consensus_innovation_and_ima(initial_local_models, K, datapoints, matrix)
    consensus_innovation_MSE = {'total': total_error}
    return consensus_innovation_MSE


iteration = 1000
weight = 2
B, weight_vec, true_labels, datapoints = get_sbm_2blocks_data(weight, pin=0.5, pout=0.5, is_torch_model=False)
all_degrees = [datapoint['degree'] for datapoint in datapoints.values()]
adjacency_matrix = incidence_to_adjacency(B)

degrees = [sum(row) for row in adjacency_matrix]
new_matrix = create_a_matrix(adjacency_matrix, degrees)

num_tries = 1
num_cores = multiprocessing.cpu_count()

def fun(matrix):
    samplingset = random.sample([j for j in range(200)], k=int(0.8 * 200))
    return get_consensus_innovation_MSE(iteration, datapoints, samplingset, matrix)

results = Parallel(n_jobs=num_cores)(delayed(fun)(new_matrix) for i in range(num_tries))

consensus_innovation_scores = defaultdict(list)
for result in results:
    consensus_innovation_scores['norm1'].append(result)

total_values = [item['total'] for item in consensus_innovation_scores['norm1']]

last_100_data = np.copy(total_values[0][900:1000])

print(f'consensus + innovation:',
      '\n mean total MSE:', np.mean(last_100_data),
      '\n std_dev total MSE:', np.std(last_100_data))

x_total = np.arange(len(total_values[0]))
plt.semilogy(x_total, np.mean(total_values, axis=0), label='total')
plt.title('Train')
plt.show()
plt.close()
