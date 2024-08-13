import numpy as np
import random
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

np.random.seed(0)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def consensus_innovation(weights, datapoints, adj_matrix, learning_rate=0.1, lambda_reg=0):
    num_nodes = adj_matrix.shape[0]
    num_features = weights.shape[1]

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
        # print(f"Round {round + 1}/{num_rounds}")

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

    # print(global_model)
    # print('----------------------------------------')
    # print(updated_weights)
    # print('----------------------------------------')
    # print(agg_weights)
    return iteration_scores, global_model, updated_weights


def generate_data(num_clusters, cluster_sizes, m=1, n=2, noise_sd=0):
    num_clusters = len(cluster_sizes)
    num_nodes = sum(cluster_sizes)

    # 生成 cluster_labels 列表，表示每个节点所属的 cluster
    cluster_labels = np.concatenate([
        np.full(size, i) for i, size in enumerate(cluster_sizes)
    ])

    datapoints = {}
    true_labels = []

    # 手动设置每个 cluster 的权重
    W = [
        np.array([2, 2]),  # Cluster 1 的权重
        np.array([-2, 2])  # Cluster 2 的权重
    ]

    for i in range(num_nodes):
        cluster = cluster_labels[i]
        features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
        label = np.dot(features, W[cluster]) + np.random.normal(0, noise_sd)
        true_labels.append(label)

        datapoints[i] = {
            'features': features,
            'label': label,
        }

    # 创建邻接矩阵
    adjacency_matrix = create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=0.5)

    return adjacency_matrix, datapoints, cluster_labels


def create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=0.01):
    num_nodes = len(cluster_labels)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    num_clusters = len(np.unique(cluster_labels))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cluster_labels[i] == cluster_labels[j]:
                if np.random.rand() < p_in:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < p_out:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

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


def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
    num_clients = 200
    num_features = 2  # Adjust this to match your specific model's requirements
    temp = np.zeros(num_features)  # Initialize with correct dimensions for 1D weights
    initial_local_models = np.stack([temp] * num_clients, axis=0)  # Create an array of shape (num_clients, num_features)
    total_error, _, _ = federated_learning_with_consensus_innovation_and_ima(initial_local_models, K, datapoints, matrix)
    consensus_innovation_MSE = {'total': total_error}
    return consensus_innovation_MSE


def re_cluster(weights, num_clusters):
    # 使用PCA找到数据的主轴
    pca = PCA(n_components=2)
    pca.fit(weights)
    principal_axis = pca.components_[0]

    # 将数据点投影到主轴上
    projections = np.dot(weights, principal_axis)

    # 找到投影值的最小值和最大值对应的点
    leftmost_point = weights[np.argmin(projections)]
    rightmost_point = weights[np.argmax(projections)]

    # 计算每个点到最左点和最右点的距离
    dist_to_leftmost = np.linalg.norm(weights - leftmost_point, axis=1)
    dist_to_rightmost = np.linalg.norm(weights - rightmost_point, axis=1)

    # 根据距离进行聚类，距离最小的那个簇
    labels = np.where(dist_to_leftmost < dist_to_rightmost, 0, 1)

    # 可视化聚类结果
    # plt.figure(figsize=(10, 6))
    # plt.scatter(weights[:, 0], weights[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.6, edgecolor='k')
    # plt.scatter([leftmost_point[0], rightmost_point[0]], [leftmost_point[1], rightmost_point[1]],
    #             c='red', marker='x', s=100, label='Extreme Points')
    # plt.plot([leftmost_point[0], rightmost_point[0]],
    #          [leftmost_point[1], rightmost_point[1]], 'b--', label='Principal Axis')
    # plt.title("Re-clustered Points with PCA Principal Axis")
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.legend()
    # plt.show()
    # plt.close()

    return labels


def optimize_and_recluster(num_clusters, cluster_size, iteration, num_reclusters, learning_rate=0.1, lambda_reg=0):
    adjacency_matrix, datapoints, cluster_labels = generate_data(num_clusters, cluster_size)
    degrees = np.sum(adjacency_matrix, axis=1)
    new_matrix = create_a_matrix(adjacency_matrix, degrees)

    num_cores = multiprocessing.cpu_count()

    mean_mse_list = []
    std_mse_list = []
    last_reclustering_scores = []

    for i in range(num_reclusters):
        def fun(matrix):
            samplingset = random.sample([j for j in range(len(datapoints))], k=int(0.8 * len(datapoints)))
            return get_consensus_innovation_MSE(iteration, datapoints, samplingset, matrix)

        results = Parallel(n_jobs=num_cores)(delayed(fun)(new_matrix) for i in range(1))

        consensus_innovation_scores = defaultdict(list)
        for result in results:
            consensus_innovation_scores['norm1'].append(result)

        total_values = [item['total'] for item in consensus_innovation_scores['norm1']]
        last_100_data = np.copy(total_values[0][900:1000])

        mean_mse = np.mean(last_100_data)
        std_mse = np.std(last_100_data)

        if mean_mse < 100 and std_mse < 100:
            mean_mse_list.append(mean_mse)
            std_mse_list.append(std_mse)

        print(f'consensus + innovation iteration {i}:',
              '\n mean total MSE:', mean_mse,
              '\n std_dev total MSE:', std_mse)

        if i == num_reclusters - 1:
            last_reclustering_scores = total_values[0]

        num_clients = 200
        num_features = 2  # Adjust this to match your specific model's requirements
        temp = np.zeros(num_features)  # Initialize with correct dimensions for 1D weights
        initial_local_models = np.stack([temp] * num_clients,
                                        axis=0)  # Create an array of shape (num_clients, num_features)
        _, _, weights = federated_learning_with_consensus_innovation_and_ima(initial_local_models, iteration, datapoints,
                                                                              new_matrix)

        cluster_labels = re_cluster(weights, num_clusters)
        # new_matrix = create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=(0.9**i)*0.01)
        new_matrix = create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=0)
        degrees = np.sum(new_matrix, axis=1)
        new_matrix = create_a_matrix(new_matrix, degrees)

    return new_matrix, weights, mean_mse_list, std_mse_list, last_reclustering_scores


# Parameters
num_clusters = 2
cluster_size = [100, 100]
iteration = 1000
num_reclusters = 100

new_matrix, final_weights, mean_mse_list, std_mse_list, last_reclustering_scores = optimize_and_recluster(
    num_clusters, cluster_size, iteration, num_reclusters)

count = 0
for i in range(len(new_matrix[0])):
    for j in range(len(new_matrix[1])):
        if new_matrix[i][j] == 0:
            count += 1
print(count)
# Plot adjacency matrix heatmap
plt.figure(figsize=(8, 6))
plt.imshow(new_matrix, cmap='binary', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix Heatmap after re-clustering')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()

# Plot mean total MSE and std_dev total MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.semilogy(range(len(mean_mse_list)), mean_mse_list, marker='o', label='Mean Total MSE')
# plt.plot(range(len(mean_mse_list)), mean_mse_list, marker='o', label='Mean Total MSE')
plt.xlabel('Reclustering Iterations')
plt.ylabel('Mean Total MSE')
plt.title('Mean Total MSE over Reclustering Iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.semilogy(range(len(std_mse_list)), std_mse_list, marker='o', label='Std Dev Total MSE')
# plt.plot(range(len(std_mse_list)), std_mse_list, marker='o', label='Std Dev Total MSE')
plt.xlabel('Reclustering Iterations')
plt.ylabel('Std Dev Total MSE')
plt.title('Std Dev Total MSE over Reclustering Iterations')
plt.legend()

plt.tight_layout()
plt.show()

# Plot MSE vs iteration for the final reclustering
plt.figure(figsize=(8, 6))
plt.semilogy(range(len(last_reclustering_scores)), last_reclustering_scores)
# plt.plot(range(len(last_reclustering_scores)), last_reclustering_scores)
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('MSE vs Iteration for the Final Reclustering')
plt.show()