import numpy as np


def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def standard_kernel_density_estimator(data, query_points, h):
    m, d = query_points.shape

    kde_values = np.zeros((m, d))

    for i in range(m):
        for j in range(d):
            diff = data[:, j] - query_points[i, j]
            scaled_diff = diff / h
            kernel_vals = gaussian_kernel(scaled_diff)
            kde_values[i, j] = np.sum(kernel_vals) / (m * h)

    return kde_values


def modified_kernel_density_estimator(data, query_points, h, s):
    m, d = query_points.shape

    kde_values = np.zeros((m, d))

    for i in range(m):
        for j in range(d):
            diff = data[:, j] - query_points[i, j]
            scaled_diff = diff / (h * s[:, j])
            kernel_vals = gaussian_kernel(scaled_diff)
            kde_values[i, j] = np.sum(kernel_vals) / (m * h * s[i, j])

    return kde_values


def kernel_density_estimator(data, query_points):
    h = 0.3
    c = 0.5

    kde_values = standard_kernel_density_estimator(data, query_points, h)

    geometric_means = np.exp(np.mean(np.log(kde_values), axis=0))
    s = (kde_values / geometric_means) ** (-c)

    modified_kde_values = modified_kernel_density_estimator(data, query_points, h, s)

    return modified_kde_values
