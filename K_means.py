import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def compute_WCSS(data, centroids, labels):
    wcss = 0
    for i, point in enumerate(data):
        cluster_idx = int(labels[i])
        center = centroids[cluster_idx]
        wcss += np.sum((point - center) ** 2)
    return wcss  

class MyK_means:
    def __init__(self,k=4,max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.clusters = {}
        self.centroids = []

    def fit(self,data):
        self.intialize_centroids(data)
        for i in range(self.max_iter):
            self.assign_clusters(data)
            previous_centroids = np.copy(self.centroids)
            self.update_centroid()

            if np.allclose(self.centroids,previous_centroids,rtol=1e-4):
                print("well done")
                break


    def predict(self,data):
        labels = []
        for point in data:
            distances = [self._dis(point,centroid) for centroid in self.centroids]
            min_dis_index = np.argmin(distances)
            labels.append(min_dis_index)

        return np.array(labels)
                    

    def intialize_centroids(self,data):
        random_indecies = np.random.choice(data.shape[0],self.k,replace=False)
        self.centroids = data[random_indecies]

    def assign_clusters(self,data):
        self.clusters = {i : []  for i in range(self.k)}
        for point in data:
            distances_from_centroids = [self._dis(point,centroid) for centroid in self.centroids]
            min_index = np.argmin(distances_from_centroids)
            self.clusters[min_index].append(point)

    def _dis(self,point,centroid):
        return np.sum((point-centroid)**2)      

    def update_centroid(self):
        for idx,points in self.clusters.items():
            self.centroids[idx] = np.mean(points,axis=0)


if __name__ == '__main__':
    data, true_labels = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.0)

    #elbow method to get the perfect cluster number 
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = MyK_means(k=k)
        kmeans.fit(data)
        labels = kmeans.predict(data)
        wcss.append(compute_WCSS(data,np.array(kmeans.centroids),labels))  # inertia_ = WCSS


    plt.plot(K_range, wcss, marker='o')
    plt.title("Elbow Method (WCSS vs k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.show()
    

    # k = 4

    kmeans = MyK_means(k=4,max_iter=100)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    cluster_centers = np.array(kmeans.centroids)

    print("Cluster centers:\n",cluster_centers)
    print("Labels:\n", labels)



    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(data[:, 0], data[:, 1], c=true_labels,
                  s=50, cmap='viridis', marker='o', edgecolors='w', linewidth=0.5)
    ax[0].set_title('Original Data')



    ax[1].scatter(data[:, 0], data[:, 1], c=labels,
                  s=50, cmap='viridis', marker='o', edgecolors='w', linewidth=0.5)

    ax[1].scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X')
    ax[1].set_title('KMeans Clustering')


    plt.tight_layout()

    plt.show()
