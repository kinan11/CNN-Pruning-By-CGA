import numpy as np
from sklearn.datasets import load_iris
from kernel_density_estimator import kernel_density_estimator


# Załaduj zbiór danych Iris
iris = load_iris()
data = iris.data

# Punkty do oszacowania gęstości
query_points = np.random.randn(data.shape[0], data.shape[1])  # Losowe punkty do oszacowania

kde_values = kernel_density_estimator(data, query_points)

# Wyświetlenie oszacowań gęstości dla każdej cechy w każdej próbce
for i in range(len(kde_values)):
    print(f"Data Point {i + 1}: Estimated Densities = {kde_values[i]}")
