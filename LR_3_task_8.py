import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

# Get data
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# Define initial clusters using KMeans
kmeans_clustering = KMeans(n_clusters=Y.max() + 1, init='k-means++', n_init=10, max_iter=300,
                           tol=0.0001, verbose=0, random_state=None, copy_x=True)
kmeans_clustering.fit(X)
predicted_labels = kmeans_clustering.predict(X)

print("Number of clusters: 3, Number of initializations: 10, Maximum iterations: 300, Tolerance: 0.0001, Verbose: 0, "
      "Random state: None, Copy data: True")
print(predicted_labels)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, s=50, cmap='viridis')
cluster_centers = kmeans_clustering.cluster_centers_
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()


def find_custom_clusters(data, num_clusters, random_seed=2):
    # Randomly select initial cluster centers
    random_generator = np.random.RandomState(random_seed)
    initial_centers_indices = random_generator.permutation(data.shape[0])[:num_clusters]
    custom_centers = data[initial_centers_indices]

    while True:
        # Assign labels based on the nearest center
        custom_labels = pairwise_distances_argmin(data, custom_centers)
        # Find new centers at the mean of the points
        new_custom_centers = np.array([data[custom_labels == i].mean(0) for i in range(num_clusters)])
        # Check for convergence
        if np.all(custom_centers == new_custom_centers):
            break
        custom_centers = new_custom_centers
    return custom_centers, custom_labels


print("Using find_custom_clusters():")
custom_centers, custom_labels = find_custom_clusters(X, 3)
print("Number of clusters: 3, Random seed: 2")
plt.scatter(X[:, 0], X[:, 1], c=custom_labels, s=50, cmap='viridis')
plt.show()

custom_centers, custom_labels = find_custom_clusters(X, 3, random_seed=0)
print("Number of clusters: 3, Random seed: 0")
plt.scatter(X[:, 0], X[:, 1], c=custom_labels, s=50, cmap='viridis')
plt.show()

kmeans_labels = KMeans(3, random_state=0).fit_predict(X)
print("Number of clusters: 3, Random seed: 0")
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=50, cmap='viridis')
plt.show()
