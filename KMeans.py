import numpy as np
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
from collections import Counter
from scipy.spatial.distance import cdist
import random as rd

class myKMeans:
    def __init__(self, n_clusters, iters):
        """
        KMeans Class constructor.
  
        Args:
          n_clusters (int) : Number of clusters used for partitioning.
          iters (int) : Number of iterations until the algorithm stops.
  
        """
        self.n_clusters = n_clusters
        self.iters = iters
        
    def kmeans_plus_plus(self, X, n_clusters):
        """
        My implementation of the KMeans++ initialization method for computing the centroids.
  
        Args:
            X (ndarray): Dataset samples
            n_clusters (int): Number of clusters
  
        Returns:
            centroids (ndarray): Initial position of centroids
        """
        # Assign the first centroid to a random sample from the dataset.
        idx = rd.randrange(len(X))
        centroids = [X[idx]]
  
        # For each cluster
        for _ in range(1, n_clusters):
  
            # Get the squared distance between that centroid and each sample in the dataset
            squared_distances = np.array([min([np.inner(centroid - sample,centroid - sample) for centroid in centroids]) for sample in X])
  
            # Convert the distances into probabilities that a specific sample could be the center of a new centroid
            proba = squared_distances / squared_distances.sum()
  
            for point, probability in enumerate(proba):
                # The farthest point from the previous computed centroids will be assigned as the new centroid as it has the highest probability.
                if probability == proba.max():
                    centroid = point
                    break
  
            centroids.append(X[centroid])
  
        return np.array(centroids)
  
    def find_closest_centroids(self, X, centroids):
        """
        Computes the distance to the centroids and assigns the new label to each sample in the dataset.
  
        Args:
            X (ndarray): Dataset samples  
            centroids (ndarray): Number of clusters
  
        Returns:
            idx (ndarray): Closest centroids for each observation
  
        """
  
        # Set K as number of centroids
        K = centroids.shape[0]
  
        # Initialize the labels array to 0
        label = np.zeros(X.shape[0], dtype=int)
  
        # For each sample in the dataset
        for sample in range(len(X)):
            distance = []
            # Take every centroid
            for centroid in range(len(centroids)):
                # Compute Euclidean norm between a specific sample and a centroid
                norm = np.linalg.norm(X[sample] - centroids[centroid])
                distance.append(norm)
  
            # Assign the closest centroid as it's label
            label[sample] = distance.index(min(distance))
  
        return label
  
    def compute_centroids(self, X, idx, K):
        """
        Returns the new centroids by computing the mean of the data points assigned to each centroid.
  
        Args:
            X (ndarray): Dataset samples 
            idx (ndarray): Closest centroids for each observation 
            K (int): Number of clusters
  
        Returns:
            centroids (ndarray): New centroids computed
        """
  
        # Number of samples and features
        m, n = X.shape
  
        # Initialize centroids to 0
        centroids = np.zeros((K, n))
  
        # For each centroid
        for k in range(K):   
            # Take all samples assigned to that specific centroid
            points = X[idx == k]
            # Compute their mean
            centroids[k] = np.mean(points, axis=0)
  
        return centroids
  
    def fit_predict(self, X):
        """
        My implementation of the KMeans algorithm.
  
        Args:
            X (ndarray): Dataset samples
  
        Returns:
            centroids (ndarray):  Computed centroids
            labels (ndarray):     Predicts for each sample in the dataset.
        """
        # Number of samples and features
        m, n = X.shape
  
        # Compute initial position of the centroids
        initial_centroids = self.kmeans_plus_plus(X, self.n_clusters)
  
        centroids = initial_centroids   
        labels = np.zeros(m)
        
        prev_centroids = centroids
  
        # Run K-Means
        for i in range(self.iters):
            # For each example in X, assign it to the closest centroid
            labels = self.find_closest_centroids(X, centroids)
  
            # Given the memberships, compute new centroids
            centroids = self.compute_centroids(X, labels, self.n_clusters)
            
            # Check if centroids stopped changing positions
            if centroids.tolist() == prev_centroids.tolist():
                print(f'K-Means converged at {i+1} iterations')
                break
            else:
                prev_centroids = centroids
  
        return labels, centroids

input = input("CSV file name: ")
data = pd.read_csv(input)

if input == 'adult.csv':
    X = data.iloc[:, :14].values 
    y = np.where(data['income'] == 1, 1, 0)  
    print(X)
else:
    X = data.iloc[:, 1:].values 
    y = np.where(data['Feature 1'] == 1, 1, 0)

for i in range(30, 90, 20):
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=(i/100), random_state=42)
    # Step 1: Train k-means on X_train
    kmeans = myKMeans(2, 50)
    mykmeans_labels, mykmeans_centers = kmeans.fit_predict(X_train)
    # Step 2: Determine majority label for each cluster, cluster labels will either be [0,1] or [1,0]
    cluster_labels = np.zeros(2)
    for cluster_idx in range(2):
        # Get indices of X_train points in this cluster
        # Get the most common y_train label in this cluster using the indices of all the 0s or 1s where in the kmeans label
        cluster_points_idx = np.where(mykmeans_labels == cluster_idx)
        common_label = Counter(y_train[cluster_points_idx]).most_common(1)[0][0]
        cluster_labels[cluster_idx] = common_label

    # Step 3: Classify X_test points
    test_labels = []
    for test_point in X_test:
        # Compute current point's distances to each centroid and put in a list, 
        # Get index of closest centroid from this current test point, 
        # Assign label of closest centroid to the test label, basically saying this point here is the label of this 
        distances = cdist([test_point], mykmeans_centers, 'euclidean')
        closest_centroid_idx = np.argmin(distances)
        test_labels.append(cluster_labels[closest_centroid_idx])

    print(f"Test size: {i/100}, X value: {100-i}")
    accuracy = metrics.accuracy_score(y_test, test_labels)
    print("Accuracy:", accuracy)
#     precision = precision_score(y_test, mykmeans_labels)
#     recall = recall_score(y_test, mykmeans_labels)
#     f1 = f1_score(y_test, mykmeans_labels)

#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1-Score:", f1)
#     print()

# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle(f'Model Evaluation Metrics: {input}', fontsize=16)

# # Confusion Matrix
# cm = confusion_matrix(y_test, predictions)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['income < 50k', 'income >= 50k'] if input == 'adult.csv' else ['Not Feature 1', 'Feature 1'],
#             yticklabels=['income < 50k', 'income >= 50k'] if input == 'adult.csv' else ['Not Feature 1', 'Feature 1'],
#             ax=axs[0, 0])
# axs[0, 0].set_xlabel('Predicted Labels')
# axs[0, 0].set_ylabel('True Labels')
# axs[0, 0].set_title('Confusion Matrix')

# # Precision-Recall Curve
# precision, recall, thresholds = precision_recall_curve(y_train, predictions)
# axs[0,1].plot(recall, precision, marker='o')
# axs[0,1].set_title('Precision-Recall Curve')
# axs[0,1].set_xlabel('Recall')
# axs[0,1].set_ylabel('Precision')
# axs[0,1].grid()

# # ROC Curve
# axs[1,0].plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# fpr, tpr, _ = roc_curve(y_train, predictions)
# axs[1,0].plot(fpr, tpr, marker='.', label='Logistic')
# axs[1,0].set_xlabel('False Positive Rate')
# axs[1,0].set_ylabel('True Positive Rate')
# axs[1,0].legend()

# plt.show()
