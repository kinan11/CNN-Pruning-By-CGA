import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from CGA import complete_gradient_algorithm
from calaculate_h import calculate_list_of_h
from cluster_CGA import cluster_algorithm


def main():
    # iris = load_iris()
    # data = iris.data
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    data = np.array([[1.1, 3], [1.33, 3.33], [1.1, 3.1], [1.1, 3.2], [1.7, 7]])
    x = complete_gradient_algorithm(data)

    cluster_algorithm(x)


if __name__ == "__main__":
    main()
