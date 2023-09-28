import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

#generate 2D classification dataset
X, y = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=23)

class KMeansClustering:

    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    @staticmethod
    def eucledian_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids-data_point)**2, axis=1))
    
    def fit(self, X, max_iterations = 200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iterations):
            y = []

            for data_point in X:
                distances = KMeansClustering.eucledian_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                # print(cluster_num)
                y.append(cluster_num)
            y = np.array(y)
            cluster_indices = []
            # print(y)
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers=[]
            for i, indices in enumerate(cluster_indices):
                if any(indices) == False:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)
        
        return y
    



random_points = np.random.randint(0,100,(100,2))
# print(random_points[:5])
X = random_points
kmeans = KMeansClustering()
labels = kmeans.fit(X)
# print(labels)

plt.scatter(X[:,0], X[:,1], c = labels)
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1], c = range(len(kmeans.centroids)), marker="*", s=200)

plt.show()