import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot

# Define the LogisticRegression class
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.num_iterations):
            model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(model)

            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)
    
    def accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        accuracy = correct_predictions / len(y_true)
        return accuracy

input = input("CSV file name: ")
data = pd.read_csv(input)

# Separate features and target
if input == 'adult.csv':
    X = data.drop(columns=['income']).values
    y = np.where(data['income'] == 1, 1, 0)  
else:
    X = data.iloc[:, 1:].values  # Features (all columns except the first one)
    y = np.where(data['Feature 1'] == 1, 1, 0)  # Binary target variable from Feature 1

for i in range(30, 90, 20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(i/100), random_state=42)

    # Standardize numerical columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
    X_test = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)

    # Predict on the training set
    predictions = model.predict(X_test)

    print(f"Test size: {i/100}, X value: {100-i}")
    print("Predictions:", predictions)

    accuracy = model.accuracy(y_test, predictions)
    print("Accuracy:", accuracy)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Model Evaluation Metrics: {input}', fontsize=16)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not income', 'income'] if input == 'adult.csv' else ['Not Feature 1', 'Feature 1'],
            yticklabels=['Not income', 'income'] if input == 'adult.csv' else ['Not Feature 1', 'Feature 1'],
            ax=axs[0, 0])
axs[0, 0].set_xlabel('Predicted Labels')
axs[0, 0].set_ylabel('True Labels')
axs[0, 0].set_title('Confusion Matrix')

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, predictions)
axs[0,1].plot(recall, precision, marker='o')
axs[0,1].set_title('Precision-Recall Curve')
axs[0,1].set_xlabel('Recall')
axs[0,1].set_ylabel('Precision')
axs[0,1].grid()


# ROC Curve
axs[1,0].plot([0, 1], [0, 1], linestyle='--', label='No Skill')
fpr, tpr, _ = roc_curve(y_test, predictions)
axs[1,0].plot(fpr, tpr, marker='.', label='Logistic')
axs[1,0].set_xlabel('False Positive Rate')
axs[1,0].set_ylabel('True Positive Rate')
axs[1,0].legend()

# plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust space for the main title
plt.show()