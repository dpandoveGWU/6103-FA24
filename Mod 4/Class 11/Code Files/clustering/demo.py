# %% Standard packages
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# %%
# Fabricate data
X, y = make_blobs(n_samples = 150, 
    n_features = 2, 
    centers = 3, 
    cluster_std = 0.5, 
    shuffle = True, 
    random_state = 0)

# %%
# Plot the data
plt.scatter(X[:,0], X[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.xlabel('X0')
plt.ylabel('X1')
plt.grid()
plt.show()

# %%
# Create an instance of KMeans class
km = KMeans(n_clusters = 3,
           init = 'random',
           max_iter = 300,
           tol = 1e-04,
           random_state = 0)

# %%
# predict k-means classes
y_km = km.fit_predict(X)


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

# %%
# Clustering may fail sometimes
# Sample two interleaving half moons
X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])

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
