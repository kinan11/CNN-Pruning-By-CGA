import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from CGA import complete_gradient_algorithm
from cluster_CGA import cluster_algorithm


def main():
    # iris = load_iris()
    # data = iris.data
    data = np.array([[1.1], [0.1], [2.1], [1.12], [1.7]])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    x = complete_gradient_algorithm(data)

    cluster_algorithm(x)


if __name__ == "__main__":
    main()
