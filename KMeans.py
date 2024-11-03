import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def euclidean(data, centroids): # Return euclidean distances between a point & a dataset
    return np.sqrt(np.sum((centroids - data)**2, axis=1))

class Kmeans:
    def __init__(self, K=2, max_iter=300):
        self.K = K
        self.centroids = None
        self.max_iter = max_iter

    def fit(self, X):
        np.random.seed(42)

        self.centroids = np.random.uniform(np.amin(X, axis=0),np.amax(X, axis=0), size=(self.K, X.shape[1]))

        for _ in range(self.max_iter):
            y = []
            for data_point in X:
                distances = euclidean(data_point, self.centroids) #list of distance from datapoint to all centroids 
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)
            cluster_indices = []

            for i in range(self.K):
                cluster_indices.append(np.argwhere(y==i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

        return y

input = input("CSV file name: ")
data = pd.read_csv(input)

# Separate features and target
if input == 'adult.csv':
    X = data.iloc[:, :13].values 
    y = np.where(data['income'] == 1, 1, 0)  
else:
    X = data.iloc[:, 1:].values 
    y = np.where(data['Feature 1'] == 1, 1, 0)

for i in range(30, 90, 20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(i/100), random_state=42)

    # Standardize numerical columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
    X_test = scaler.transform(X_test)

    random_points = np.random.randint(0, 1000,(1000, 2))
    random_points2 = pd.DataFrame(random_points)
    y2 = random_points2.iloc[:, 1:].values
    X_train2, X_test2, y_train2, y_test2 = train_test_split(random_points2, y2, test_size=(i/100), random_state=42)

    # Train the model
    model = Kmeans(2,100)
    pred = model.fit(X_train2)
    # print(model.centroids)

    plt.scatter(random_points[:,0], random_points[:,1], c=pred)
    plt.scatter(model.centroids[:,0], model.centroids[:,1], c=range(len(model.centroids))
                ,marker='o', s = 200)
    plt.show()