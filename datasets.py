from sklearn.datasets import make_blobs, make_moons
import numpy as np
import matplotlib.pyplot as plt

class datasetManager():
    def __init__(self,n_samples = 300,n_features = 2,centers = 3,random_state=42) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.centers = centers
        self.random_state = random_state

    def blobs(self,debug = False):
        X_true, _ = make_blobs(n_samples=self.n_samples, n_features=self.n_features, centers=self.centers, random_state=self.random_state )
        Y_true = np.ones(len(X_true))

        n_outliers = 20
        np.random.seed(self.random_state)
        X_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, self.n_features))
        y_outliers = -np.ones(len(X_outliers))

        X = np.vstack([X_true, X_outliers])
        y_true = np.concatenate([Y_true, y_outliers])

        if debug:
            plt.figure(figsize=(10, 6))
            plt.title("Generated make_blobs samples with added outliers")
            plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')
            plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')

        return X,y_true
    
    def moons(self, debug=False):
        X_true, _ = make_moons(n_samples=self.n_samples, noise=0.1, random_state=self.random_state)
        Y_true = np.ones(len(X_true))

        n_outliers = 20
        np.random.seed(self.random_state)
        X_outliers = np.random.uniform(low=-2, high=3, size=(n_outliers, 2))
        y_outliers = -np.ones(len(X_outliers))

        X = np.vstack([X_true, X_outliers])
        y_true = np.concatenate([Y_true, y_outliers])

        if debug:
            plt.figure(figsize=(10, 6))
            plt.title("Generated make_moons samples with added outliers")
            plt.scatter(X[:, 0], X[:, 1], c='white', s=20, edgecolor='k')
            plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
            plt.show()

        return X, y_true