import numpy as np
from sklearn.cluster import KMeans

# Dataset
X = np.array([
    [1, 2], [2, 3], [3, 2],
    [8, 8], [9, 8], [8, 9],
    [15, 15], [16, 15], [15, 16]
])

# Create and train K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

# Display cluster results
print("Cluster labels:", model.labels_)
print("Cluster centers:")
print(model.cluster_centers_)

# Predict cluster for a new point
print("New point cluster:", model.predict([[2, 2]])[0])
