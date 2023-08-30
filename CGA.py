import numpy as np

from kernel_density_estimator import kernel_density_estimator


def calculate_d(x):
    dist = 0
    for i in range(x.shape[0] - 1):
        for j in range(i+1, x.shape[0]):
            dist += np.linalg.norm(x[i] - x[j])
    return dist


def complete_gradient_algorithm(data, h):
    num_iterations = 10
    x = data.copy()
    f_value_prev = []
    b = (h ** 2) / (data.shape[0] + 2)
    d0 = calculate_d(data)
    alpha = 0.001

    for iteration in range(num_iterations):
        dk_prev = calculate_d(x)
        f_value_curr = kernel_density_estimator(x, x)

        if iteration > 0:
            for i in range(data.shape[0]):
                f_gradient = np.gradient(f_value_curr[i], f_value_prev[i])
                x[i] += b * (f_gradient / f_value_curr[i])

            dk = calculate_d(x)
            if abs(dk - dk_prev) <= alpha * d0:
                break

        f_value_prev = f_value_curr.copy()

    return x
