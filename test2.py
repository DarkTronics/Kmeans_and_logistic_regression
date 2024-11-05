import numpy as np
import pandas as pd

class CustomKMeans:
    def __init__(self, k, X):
        '''
        k: number of clusters
        centroids:  initial value of centres. 
                    np.array of shape (k, n_features)
        X:  dataset of m points and n_features. 
            array of shape (m, n_features)
        '''
        self.k = k
        self.centroids = np.random.uniform(np.amin(X, axis=0),np.amax(X, axis=0), size=(self.k, X.shape[1]))
        self.X = X
        self.dim = X.shape[1]
        self.index = {i:[] for i in range(len(self.centroids))}


    def get_squared_distance_from_centroid(self, centroid):
        '''
        compute the squared distance of a 
        centroid from each point in X
        '''
        return np.sum(np.square(self.X-centroid), axis=1)


    def get_cluster_variances(self):
        '''
        For each centroid, 
        get get_squared_distance_from_centroid
        '''
        intra_cluster_variances = []
        for i in range(self.k):
            centroid = self.centroids[i]
            distance = self.get_squared_distance_from_centroid(centroid)
            intra_cluster_variances.append(
                                    distance.reshape(-1,1)
                                )
        return np.hstack(intra_cluster_variances)


    def update_centroids(self, min_variance_clusters):
        '''
        Update the index based on closest centroid 
        for a point and then update the centroid
        based on this updated index
        '''
        for i in range(self.k):
            self.index[i] = np.where(min_variance_clusters==i)[0]
            self.centroids[i] = np.mean(
                                    self.X[self.index[i]], 
                                    axis=0
                                )

    
    def update(self):
        '''
        Update step in the KMeans algorithm.
        '''
        intra_cluster_variances = self.get_cluster_variances()
        min_variance_clusters = np.argmin(
                                    intra_cluster_variances, 
                                    axis=1
                                )
        self.update_centroids(min_variance_clusters)
        


input = input("CSV file name: ")
data = pd.read_csv(input)
k = 2
# Separate features and target
if input == 'adult.csv':
    X = data.iloc[:, :13].values 
    y = np.where(data['income'] == 1, 1, 0)  
    print(X)
else:
    X = data.iloc[:, 1:].values 
    y = np.where(data['Feature 1'] == 1, 1, 0)

model = CustomKMeans(k, X)