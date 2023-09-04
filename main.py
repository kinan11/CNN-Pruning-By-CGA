from sklearn.datasets import load_iris
from CGA import complete_gradient_algorithm
from cluster_CGA import cluster_algorithm


def main():
    iris = load_iris()
    data = iris.data

    h = 0.3
    x = complete_gradient_algorithm(data, h)
    print(x)
    cluster_algorithm(x)


if __name__ == "__main__":
    main()
