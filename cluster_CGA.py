import math
import numpy as np
import statistics

from calaculate_h import calculate_list_of_h
from kernel_density_estimator import kernel_density_estimator


def calculate_distances(data):
    distances = []
    for i in range(data.shape[0] - 1):
        for j in range(i + 1, data.shape[0]):
            distances.append(np.linalg.norm(data[i] - data[j]))

    return distances


def calculate_x_d(data, h):
    distances = calculate_distances(data)
    max_d = max(distances)
    sigma = statistics.stdev(distances)  # odchylenie standardowe
    d = (math.floor(100 * max_d) - 1)
    converted_d = np.array([[x * 0.01 * sigma] for x in range(math.floor(d))])
    h = calculate_list_of_h(converted_d)
    kde_d, s = kernel_density_estimator(converted_d, h)
    x_d = kde_d[1]

    for i in range(1, math.floor(d) - 1):
        x_d = kde_d[i]
        if kde_d[i - 1] > kde_d[i] <= kde_d[i + 1]:
            break
    return x_d


def calculate_distance(data):
    distances = np.zeros(data.shape[0] - 1)
    for i in range(1, data.shape[0]):
        distances[i - 1] = np.linalg.norm(data[0] - data[i])
    return distances


def cluster_algorithm(data, h):
    x_d = calculate_x_d(data, h)
    clusters = []
    while data.shape[0] > 0:
        distances = calculate_distance(data)
        indexes = [0]
        for index, element in enumerate(distances):
            if element < x_d:
                indexes.append(index+1)
        if indexes:
            cluster = [data[i] for i in reversed(indexes)]
            clusters.append(cluster)
            data = np.delete(data, indexes, axis=0)

    print(x_d, len(clusters))
    print(clusters)
