#In this problem, you will generate simulated data, and then perform K-means clustering on the data. 
#1.Generate a simulated data set with 20 observations in each of  three classes (i.e. 60 observations total), and 50 variables.  
# Hint: There are a number of functions in Python that you can  use to generate data. One example is the normal() method of  the random() function in numpy; the uniform() method is another  option. Be sure to add a mean shift to the observations in each  class so that there are three distinct classes. 

#2.Perform K-means clustering of the observations with K = 3. How well do the clusters that you obtained in K-means clustering compare to the true class labels?  
# Hint: You can use the pd.crosstab() function in Python to compare the true class labels to the class labels obtained by clustering. Be careful how you interpret the results: K-means clustering  will arbitrarily number the clusters, so you cannot simply check  whether the true class labels and clustering labels are the same. 

#3. Using the StandardScaler() estimator, perform K-means clustering with K = 3 on the data after scaling each variable to have  standard deviation one. How do these results compare to those  obtained in (2)? Explain. 



# 

