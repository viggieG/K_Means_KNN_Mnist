import numpy as np

class KMeans():
    def __init__(self, n_clusters, max_iterations=100, tolerance=1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.means = None

    def fit(self, features):
        # Step 1: Initialize cluster centroids
        initial_centroids_idx = np.random.choice(features.shape[0], self.n_clusters, replace=False)
        self.means = features[initial_centroids_idx]

        for _ in range(self.max_iterations):
            old_means = self.means.copy()
            # Step 2: Assign clusters
            assignments = self._update_assignments(features)

            # Step 3: Update centroids
            self._update_means(features, assignments)

            # Check for convergence
            if np.linalg.norm(self.means - old_means) < self.tolerance:
                break

    def predict(self, features):
        return self._update_assignments(features)

    def _update_assignments(self, features):
        distances = np.linalg.norm(features[:, np.newaxis] - self.means, axis=2)
        return np.argmin(distances, axis=1)

    def _update_means(self, features, assignments):
        for i in range(self.n_clusters):
            cluster_points = features[assignments == i]
            if cluster_points.size > 0:
                self.means[i] = cluster_points.mean(axis=0)



