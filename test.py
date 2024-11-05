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

class Kmeans:
    def __init__(self) -> None:
        pass

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
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)

    # Step 2: Determine majority label for each cluster, cluster labels will either be [0,1] or [1,0]
    cluster_labels = np.zeros(kmeans.n_clusters)
    for cluster_idx in range(kmeans.n_clusters):
        # Get indices of X_train points in this cluster
        cluster_points_idx = np.where(kmeans.labels_ == cluster_idx)
        # Get the most common y_train label in this cluster using the indices of all the 0s or 1s where in the kmeans label
        common_label = Counter(y_train[cluster_points_idx]).most_common(1)[0][0]
        cluster_labels[cluster_idx] = common_label

    # Step 3: Classify X_test points
    test_labels = []
    for test_point in X_test:
        # Compute current point's distances to each centroid and put in a list
        distances = cdist([test_point], kmeans.cluster_centers_, 'euclidean')
        # Get index of closest centroid from this current test point
        closest_centroid_idx = np.argmin(distances)
        # Assign label of closest centroid to the test label, basically saying this point here is the label of this 
        test_labels.append(cluster_labels[closest_centroid_idx])

    accuracy = metrics.accuracy_score(y_test, test_labels)
    print("Accuracy:", accuracy)
#     precision = precision_score(y_train, predictions)
#     recall = recall_score(y_train, predictions)
#     f1 = f1_score(y_train, predictions)

#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1-Score:", f1)
#     print()

# fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# fig.suptitle(f'Model Evaluation Metrics: {input}', fontsize=16)

# # Confusion Matrix
# cm = confusion_matrix(y_train, predictions)
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
