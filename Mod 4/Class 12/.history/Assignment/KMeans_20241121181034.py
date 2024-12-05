#In this problem, you will generate simulated data, and then perform K-means clustering on the data. 
#1.Generate a simulated data set with 20 observations in each of  three classes (i.e. 60 observations total), and 50 variables.  
# Hint: There are a number of functions in Python that you can  use to generate data. One example is the normal() method of  the random() function in numpy; the uniform() method is another  option. Be sure to add a mean shift to the observations in each  class so that there are three distinct classes. 

#2.Perform K-means clustering of the observations with K = 3. How well do the clusters that you obtained in K-means clustering compare to the true class labels?  
# Hint: You can use the pd.crosstab() function in Python to compare the true class labels to the class labels obtained by clustering. Be careful how you interpret the results: K-means clustering  will arbitrarily number the clusters, so you cannot simply check  whether the true class labels and clustering labels are the same. 

#3. Using the StandardScaler() estimator, perform K-means clustering with K = 3 on the data after scaling each variable to have  standard deviation one. How do these results compare to those  obtained in (2)? Explain. 



# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
# Step 1: Generate simulated data
np.random.seed(42)  # Set seed for reproducibility
# %%

# Create 3 classes with 20 observations each and 50 variables
class_1 = np.random.normal(loc=0, scale=1, size=(20, 50))  # Class 1 centered at mean 0
class_2 = np.random.normal(loc=5, scale=1, size=(20, 50))  # Class 2 centered at mean 5
class_3 = np.random.normal(loc=10, scale=1, size=(20, 50))  # Class 3 centered at mean 10

# Combine data into one dataset
data = np.vstack([class_1, class_2, class_3])
true_labels = np.array([0]*20 + [1]*20 + [2]*20)  # True class labels

# %%
# Step 2: Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(data)

# Compare clustering results with true labels
comparison = pd.crosstab(true_labels, cluster_labels)


# Step 3: Scale the data and perform K-means clustering again
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

kmeans_scaled = KMeans(n_clusters=3, random_state=42)
cluster_labels_scaled = kmeans_scaled.fit_predict(scaled_data)

# Compare clustering results with true labels for scaled data
comparison_scaled = pd.crosstab(true_labels, cluster_labels_scaled)

comparison, comparison_scaled


