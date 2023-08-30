from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from CGA import complete_gradient_algorithm


def main():
    iris = load_iris()
    data = iris.data

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    h = 0.3

    print(complete_gradient_algorithm(data, h))


if __name__ == "__main__":
    main()
