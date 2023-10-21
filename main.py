import numpy as np
from sklearn.datasets import load_iris
from CGA import complete_gradient_algorithm
from calaculate_h import calculate_list_of_h
from cluster_CGA import cluster_algorithm
from kernel_density_estimator import kernel_density_estimator


def main():
    # iris = load_iris()
    # data = iris.data
    data = np.array([[3, 6], [3.33, 7], [5, 4]])
    h = calculate_list_of_h(data)
    x = complete_gradient_algorithm(data, h)
    print(x)
    cluster_algorithm(x, h)


if __name__ == "__main__":
    main()
