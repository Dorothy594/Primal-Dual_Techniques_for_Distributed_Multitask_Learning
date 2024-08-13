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
from sklearn.cluster import KMeans


np.random.seed(0)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# def consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.1, lambda_reg=0, local_iterations=100, calculate_score=False):
#     """
#     Combine federated learning with consensus innovation.
#
#     :param iterations: Number of global iterations.
#     :param datapoints: Local data for each node.
#     :param adj_matrix: Adjacency matrix for consensus.
#     :param learning_rate: Learning rate for gradient descent.
#     :param lambda_reg: Regularization parameter.
#     :param local_iterations: Number of local iterations before global update.
#     :param calculate_score: Boolean to calculate and return MSE scores.
#     :return: iteration_scores, weights
#     """
#     num_nodes = adj_matrix.shape[1]  # Number of nodes
#     weights = np.zeros((num_nodes, datapoints[0]['features'].shape[1]))  # Initialize local weights
#     iteration_scores = []
#
#     for i in range(num_nodes):
#         weights[i] = np.zeros(datapoints[i]['features'].shape[1])
#
#     for global_iter in range(iterations):
#         # Local training for each node
#         for local_iter in range(local_iterations):
#             for i in range(num_nodes):
#                 gradient = np.dot(datapoints[i]['features'].T, (np.dot(datapoints[i]['features'], weights[i]) - datapoints[i]['label']))
#                 regularization_term = lambda_reg * weights[i]
#                 gradient += regularization_term
#
#                 weights[i] -= learning_rate * gradient
#
#         # Consensus update
#         weights_new = np.copy(weights)
#
#         for _ in range(0):
#             for i in range(num_nodes):
#                 consensus_sum = np.zeros(weights[i].shape)
#                 for j in range(num_nodes):
#                     if adj_matrix[i, j] != 0:
#                         consensus_sum += adj_matrix[i, j] * weights[j]
#
#                 # weights_new[i] = consensus_sum / np.sum(adj_matrix[i])
#
#                 gradient = np.dot(datapoints[i]['features'].T,
#                                   (np.dot(datapoints[i]['features'], weights[i]) - datapoints[i]['label']))
#                 regularization_term = lambda_reg * weights[i]
#                 gradient += regularization_term
#
#                 weights_new[i] = consensus_sum - learning_rate * gradient
#
#         weights = np.copy(weights_new)
#
#         if calculate_score:
#             Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
#             true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
#             mse = mean_squared_error(true_labels.flatten(), Y_pred.flatten())
#             iteration_scores.append(mse)
#
#     print(weights)
#     plt.scatter(weights[:, 0], weights[:, 1], cmap='viridis', marker='o', alpha=0.6, edgecolor='k')
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.legend()
#     plt.show()
#     plt.close()
#
#     return iteration_scores, weights


# def consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.1, lambda_reg=0, calculate_score=False):
#     num_nodes = adj_matrix.shape[1]  # Number of nodes
#     weights = np.zeros((num_nodes, datapoints[0]['features'].shape[1]))  # Initialize local variables
#     iteration_scores = []
#
#     for i in range(num_nodes):
#         weights[i] = np.zeros(datapoints[i]['features'].shape[1])
#
#     for _ in range(iterations):
#         weights_new = np.copy(weights)
#
#         for i in range(num_nodes):
#             consensus_sum = np.zeros(weights[i].shape)
#             for j in range(num_nodes):
#                 if adj_matrix[i, j] != 0:
#                     consensus_sum += adj_matrix[i, j] * weights[j]
#
#             # Gradient computation as per federated learning
#             gradients = np.zeros(weights[i].shape)
#             for k in range(datapoints[i]['features'].shape[0]):
#                 x_k = datapoints[i]['features'][k]
#                 y_k = datapoints[i]['label'][k]
#                 gradients += (np.dot(x_k, weights[i]) - y_k) * x_k
#             gradients /= datapoints[i]['features'].shape[0]
#
#             regularization_term = lambda_reg * weights[i]
#             gradients += regularization_term
#
#             weights_new[i] = consensus_sum - learning_rate * gradients
#
#         weights = np.copy(weights_new)
#
#         if calculate_score:
#             Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
#             true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
#             mse = mean_squared_error(true_labels, Y_pred)
#             iteration_scores.append(mse)
#
#     print(weights)
#     plt.scatter(weights[:, 0], weights[:, 1], cmap='viridis', marker='o', alpha=0.6, edgecolor='k')
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.legend()
#     plt.show()
#     plt.close()
#
#     return iteration_scores, weights


def consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.005, lambda_reg=0, local_iterations=50,
                         calculate_score=False):
    # pout=0.5：lr=0.01
    # pout=0.01， lr=0.005
    num_nodes = adj_matrix.shape[1]  # Number of nodes
    weights = np.zeros((num_nodes, datapoints[0]['features'].shape[1]))  # Initialize local variables
    iteration_scores = []

    for i in range(num_nodes):
        weights[i] = np.zeros(datapoints[i]['features'].shape[1])

    for _ in range(iterations):
        weights_new = np.copy(weights)

        for i in range(num_nodes):
            # Perform multiple local iterations of federated learning
            local_weights = np.copy(weights[i])
            for _ in range(local_iterations):
                gradients = np.zeros(local_weights.shape)
                for k in range(datapoints[i]['features'].shape[0]):
                    x_k = datapoints[i]['features'][k]
                    y_k = datapoints[i]['label'][k]
                    gradients += (np.dot(x_k, local_weights) - y_k) * x_k
                gradients /= datapoints[i]['features'].shape[0]

                regularization_term = lambda_reg * local_weights
                gradients += regularization_term

                local_weights -= learning_rate * gradients

            consensus_sum = np.zeros(weights[i].shape)
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    consensus_sum += adj_matrix[i, j] * weights[j]

            weights_new[i] = consensus_sum + local_weights - weights[i]

        weights = np.copy(weights_new)

        if calculate_score:
            Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
            true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
            mse = mean_squared_error(true_labels, Y_pred)
            iteration_scores.append(mse)

    print(weights)
    plt.scatter(weights[:100, 0], weights[:100, 1], cmap='viridis', marker='o', alpha=0.6, edgecolor='k')
    plt.scatter(weights[101:, 0], weights[101:, 1], cmap='red', marker='o', alpha=0.6, edgecolor='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
    plt.close()

    return iteration_scores, weights


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

    # node_degrees = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
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

    # print(f"Global clustering coefficient: {nx.average_clustering(G)}")

    N = len(G.nodes)
    E = len(G.edges)
    # density = 2 * E / (N * (N-1))
    # print(f"Density: {density}")

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
    total_error, _ = consensus_innovation(K, datapoints, matrix, calculate_score=True)
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
# plt.plot(x_total, np.mean(total_values, axis=0), label='total')
plt.title('Train')
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
plt.imshow(new_matrix, cmap='binary', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix Heatmap')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()