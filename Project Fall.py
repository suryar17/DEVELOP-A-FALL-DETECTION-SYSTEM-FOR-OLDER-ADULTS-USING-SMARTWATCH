#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("ac.csv")


# In[3]:


df.head(10)


# In[4]:


df = df.drop(df.columns[0], axis='columns')


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20, 15))

# Subplot 1 (top)
plt.subplot(4, 1, 1)  # 2 rows, 1 column, first subplot
plt.plot(df.iloc[:, 0], color='blue')
plt.title('ax (m/s^2)')

# Subplot 2 (bottom)
plt.subplot(4, 1, 2)
plt.plot( df.iloc[:, 1], color='red')
plt.title('ay (m/s^2)')

# Subplot 3 (bottom)
plt.subplot(4, 1, 3)
plt.plot( df.iloc[:, 2], color='yellow')
plt.title('az (m/s^2)')


# Subplot 4 (bottom)
plt.subplot(4, 1, 4)  # 2 rows, 1 column, second subplot
plt.plot( df.iloc[:, 3], color='green')
plt.title('aT (m/s^2)')


# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


# In[7]:


sns.boxplot(data=df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']])
plt.show()


# In[8]:


#extract features
features = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']].values

# thresholding the output
threshold_value = 0.5

# Create binary labels (1 for fall, 0 for not fall) based on the 'aT' column
df['TrueFallLabels'] = (df['aT (m/s^2)'] > threshold_value).astype(int)

# Replace 'TrueFallLabels' with your actual ground truth labels column name
true_fall_labels = df['TrueFallLabels']


# In[9]:


df.head(50)


# In[10]:


import numpy as np

def kmeans(X, n_clusters, max_iters=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

        # Update centroids based on the mean of assigned data points
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels

def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_samples = len(true_labels)
    accuracy = correct_predictions / total_samples
    return accuracy


# In[11]:


from sklearn.metrics import silhouette_score

features = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']].values

# Standardize the features
features = (features - features.mean(axis=0)) / features.std(axis=0)

# Number of clusters (you can adjust this based on your problem)
n_clusters = 2

# Perform K-means clustering
predicted_labels = kmeans(features, n_clusters)

# Calculate silhouette score
silhouette_avg = silhouette_score(features, predicted_labels)
print(f"Silhouette Score: {silhouette_avg}")


# In[14]:


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            # Update weights and bias
            self.weights -= self.learning_rate * np.dot(X.T, predictions - y) / len(y)
            self.bias -= self.learning_rate * np.sum(predictions - y) / len(y)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Extract features and labels
X = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)', 'aT (m/s^2)']].values
y = df['TrueFallLabels']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)


# In[16]:


# Print accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f'Model Accuracy: {accuracy}')


# In[17]:


# Deep Learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

# Custom metric to calculate accuracy for regression
def accuracy(y_true, y_pred):
    threshold = 0.10  # You can adjust this threshold based on your problem
    y_pred_binary = K.cast(K.greater(y_pred, threshold), K.floatx())
    return K.mean(K.equal(y_true, y_pred_binary))

# Assuming the last column 'aT' represents the target variable
X = df.drop(columns=['aT (m/s^2)'])
y = df['aT (m/s^2)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network for regression
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with the custom accuracy metric
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[accuracy])

# Train the model
model.fit(X_train_scaled, y_train, epochs=5, batch_size=32, validation_data=(X_test_scaled, y_test))

# Evaluate the model
_, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Model Accuracy: {accuracy}')


# In[ ]:





# In[18]:


# Now, for a new sample input
sample_input = np.array([[0.037315	,-0.0497,	0.3747	,-0.314]])
sample_input_scaled = scaler.transform(sample_input)
predicted_value = model.predict(sample_input_scaled)

print(f'Predicted Value for the Sample Input: {predicted_value[0, 0]}')

from twilio.rest import Client

account_sid = 'AC4cf6adcd2b1f7695f24e87518ed0bee2'
auth_token = '8c749bad6e98bd46bb649022da279b93'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='whatsapp:+14155238886',
  body='Fall Detected Please check it once',
  to='whatsapp:+919344857514'
)

if predicted_value[0, 0] < threshold_value:
  print("Fall detected")
else:
  print("No fall")


# In[ ]:





# In[ ]:




