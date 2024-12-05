# %% Standard packages
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# %%
# Fabricate data
X, y = make_blobs(n_samples = 150, 
    n_features = 2, 
    centers = 3, #No of clusters
    cluster_std = 0.5, #Standard deviation of the clusters. A smaller value makes clusters tighter, while a larger value makes them more spread out.
    shuffle = True, 
    random_state = 0)
#make_blobs: A function to generate a dataset consisting of Gaussian-distributed clusters of points.
#X: A 2D array (features) containing the generated sample points' coordinates. 
# Each row corresponds to a data point, and each column corresponds to a feature.
#y: A 1D array (labels) with the cluster label for each point in X. 
# These labels (0, 1, 2, ...) identify which cluster the point belongs to.

# %%
# Plot the data
plt.scatter(X[:,0], X[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
#X[:, 0]: The first column of X, representing the values for the first feature (X0).
#X[:, 1]: The second column of X, representing the values for the second feature (X1).
plt.xlabel('X0')
plt.ylabel('X1')
plt.grid()
plt.show()

# %%
# Create an instance of KMeans class
km = KMeans(n_clusters = 3,
           init = 'random', #'random' means the initial centroids will be chosen randomly from the data points.
           max_iter = 300, #If the algorithm hasn't converged within 300 iterations, it will stop
           tol = 1e-04, #Convergence tolerance
           random_state = 0)

#convergence tolerance
#The algorithm stops if the change in the positions of the centroids between iterations is less than 0.0001 (or 1e-04).
#This prevents the algorithm from running unnecessarily once the solution stabilizes.

# %%
# predict k-means classes
y_km = km.fit_predict(X) #Stores the cluster labels (predicted by K-Means) for all points in X

#fit_predict is a combination of two steps:

#1. fit(X): Fits the K-Means model to the dataset X. This step involves:
#Initializing centroids based on the parameters provided (e.g., init='random').
#Iteratively updating centroids by minimizing the within-cluster variance until convergence or reaching the max_iter limit.

#2.predict(X): Assigns each data point in X to the nearest cluster centroid, effectively labeling each point with a cluster ID.

# %%
# visualize cluster predictions
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()

# %%
# Scoring
km.inertia_
#within-cluster sum of squared distances (WCSS), a key metric in clustering analysis.
#If inertia is high, it suggests that the points are not well-clustered (points are far from their cluster centroids).
#If inertia is very low, it could mean overfitting (e.g., each data point becomes its own cluster in extreme cases).


#One Output
#An inertia of 72.48 is not inherently good or bad—evaluate it relative to your dataset's characteristics and clustering goals. 
# Use the elbow method and additional metrics like the silhouette score to validate clustering quality.

# %%
# Clustering may fail sometimes
# Sample two interleaving half moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
#noise: Adds Gaussian noise to the dataset to make it slightly more complex and less perfectly separable.
#The output will be a scatter plot of 200 points arranged in two interleaved crescent shapes (moons). Noise will make the boundaries between the two moons slightly fuzzy, adding a challenge for clustering or classification algorithms.


# %%
# cluster using k-means
km = KMeans(n_clusters=2, random_state=0, n_init = 10)
y_km = km.fit_predict(X)

# visualize cluster predictions
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Centroids')
plt.legend(scatterpoints=1)
plt.grid()


# %%
