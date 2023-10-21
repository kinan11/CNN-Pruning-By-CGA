import numpy as np


def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)


def standard_kernel_density_estimator(data, h):
    m, d = data.shape

    kde_values = np.zeros(m)

    for i in range(m):
        for j in range(m):
            kernel_vals = 1
            for k in range(d):
                diff = data[i, k] - data[j, k]
                scaled_diff = diff / h[k]
                kernel_vals *= gaussian_kernel(scaled_diff)
            kde_values[i] += kernel_vals
        kde_values[i] = kde_values[i] / (m * np.prod(h))

    return kde_values


def modified_kernel_density_estimator(data, h, s):
    m, d = data.shape

    kde_values = np.zeros(m)

    for i in range(m):
        for j in range(m):
            kernel_vals = 1
            for k in range(d):
                diff = data[i, k] - data[j, k]
                scaled_diff = diff / (h[k] * s[j])
                kernel_vals *= gaussian_kernel(scaled_diff)
            kde_values[i] += kernel_vals / s[j]
        kde_values[i] = kde_values[i] / (m * np.prod(h))

    return kde_values


def kernel_density_estimator(data, h):
    c = 0.5

    kde_values = standard_kernel_density_estimator(data, h)

    geometric_means = np.exp(np.mean(np.log(kde_values), axis=0))
    s = (kde_values / geometric_means) ** (-c)

    modified_kde_values = modified_kernel_density_estimator(data, h, s)

    return modified_kde_values
