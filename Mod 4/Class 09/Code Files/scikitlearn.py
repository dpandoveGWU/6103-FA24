# %% [markdown]
# shell install scikit-learn
# https://scikit-learn.org/stable/install.html  
#
# Scikit-learn requires:
# * Python (>= 3.5)
# * NumPy (>= 1.11.0)
# * SciPy (>= 0.17.0)
# * joblib (>= 0.11)
#
# You can check what is installed on your system by shell command
# pip list 
# pip freeze
# 
# Since we already have scipy and numpy installed, simply do this
# %pip3 install -U scikit-learn
# %pip install -U scikit-learn
# OR
# %conda install scikit-learn 

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model


# %%
# First read in the datasets. 
dfpizza = pd.read_csv('Pizza.csv')

# %% [markdown]
# The dataset is clean. No missing values.  
# brand -- Pizza brand (class label)  
# id -- Sample analysed  
# mois -- Amount of water per 100 grams in the sample  
# prot -- Amount of protein per 100 grams in the sample  
# fat -- Amount of fat per 100 grams in the sample  
# ash -- Amount of ash per 100 grams in the sample  
# sodium -- Amount of sodium per 100 grams in the sample  
# carb -- Amount of carbohydrates per 100 grams in the sample  
# cal -- Amount of calories per 100 grams in the sample  


# %%
# Last time we were using the statsmodel library and the functions there, 
# let us try scikit-learn this time.

# %%
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xpizza = dfpizza[['mois', 'prot', 'fat', 'ash', 'sodium', 'carb']]
print(xpizza.head())
print(type(xpizza))

# We can try two different ways to prepare for y-data
ypizza = dfpizza['cal']
# ypizza = dfpizza[['cal']]

# print(ypizza.head())
# print(type(ypizza))

# These xdfpizza and ydfpizza are dataframes


# %%
sns.pairplot(xpizza)
# plt.title("seaborn pairplot")
plt.show()



# %%
# There are a lot of interesting relationships shown here. 
# But let us first try the basic linear regression using sklearn
fatcalfit = linear_model.LinearRegression()  # instantiate the object, with full functionality

# %%
# Fit the model
model = fatcalfit.fit( xpizza, ypizza )

# %%
# Print the coeff and intercept
model.intercept_.round(5)
model.coef_.round(5)

# %%
# .score is the generic method of evaluating models in sklearn
# for linear regression, it is R^2 value. 
# This is a simple linear model with one predictor we have.
fatcalfit.score( xpizza, ypizza )


# %% [markdown]
# # sklearn does not give p-values for each regressor's coefficient
# 
# nor other statistics such as skewness, etc. 
# The statsmodel library handle those as we 
# learned last time. You can find some get-around solutions...
# 


# %%
# Let us also do the model evaluation using train and test sets from the early on
from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(xpizza, ypizza, test_size = 0.250, random_state=333)
full_split1 = linear_model.LinearRegression() # new instance
full_split1.fit(X_train1, y_train1)
y_pred1 = full_split1.predict(X_test1)

print('score (train R^2):', full_split1.score(X_train1, y_train1)) # 0.9991344781997908
print('intercept:', full_split1.intercept_) # 6.200675408951385
print('coef_:', full_split1.coef_)  # [-0.06211495 -0.02190932  0.02823002 -0.06189928 -0.00402231 -0.02191296] 

# Use the trained model to get R^2 of the test dataset
print('score (test R^2):', full_split1.score(X_test1, y_test1)) # 0.999921891245662


# %%
# Logistic Regression
dfadmit = pd.read_csv('gradAdmit.csv')


# %%
# From last week, we used statsmodels.api
import statsmodels.api as sm  # Importing statsmodels
from statsmodels.formula.api import glm

# model = glm(formula, data, family)
modelAdmitGreLogitFit = glm(formula='admit ~ gre', data=dfadmit, family=sm.families.Binomial()).fit()
print( modelAdmitGreLogitFit.summary() )




# %% 
# Let's try logistic regression again with sklearn instead of statsmodel
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xadmit = dfadmit[['gre', 'gpa', 'rank']]
yadmit = dfadmit['admit']
print(type(xadmit))
print(type(yadmit))


# %%
x_trainAdmit, x_testAdmit, y_trainAdmit, y_testAdmit = train_test_split(xadmit, yadmit, random_state = 1)

print('x_trainAdmit type', type(x_trainAdmit))
print('x_trainAdmit shape', x_trainAdmit.shape)
print('x_testAdmit type', type(x_testAdmit))
print('x_testAdmit shape', x_testAdmit.shape)
print('y_trainAdmit type', type(y_trainAdmit))
print('y_trainAdmit shape', y_trainAdmit.shape)
print('y_testAdmit type', type(y_testAdmit))
print('y_testAdmit shape', y_testAdmit.shape)


# %%
# Logit using Scikit
from sklearn.linear_model import LogisticRegression

admitlogit = LogisticRegression()  # instantiate
admitlogit.fit( x_trainAdmit, y_trainAdmit )
print('Logit model accuracy (with the test set):', admitlogit.score(x_testAdmit, y_testAdmit))
print('Logit model accuracy (with the train set):', admitlogit.score(x_trainAdmit, y_trainAdmit))


# %%
print(admitlogit.predict(x_testAdmit))


# %%
print(admitlogit.predict_proba(x_trainAdmit[:8]))
# print(admitlogit.predict_proba(x_testAdmit[:8]))


# %%
# Classification Report
#
from sklearn.metrics import classification_report
print( classification_report( y_testAdmit, admitlogit.predict(x_testAdmit) ) )

#                         predicted 
#                   0                  1
# Actual 0   True Negative  TN      False Positive FP
# Actual 1   False Negative FN      True Positive  TP
# 
# Accuracy    = (TP + TN) / Total
# Precision   = TP / (TP + FP)
# Recall rate = TP / (TP + FN) = Sensitivity
# Specificity = TN / (TN + FP)
# F1_score is the "harmonic mean" of precision and recall
#          F1 = 2 (precision)(recall)/(precision + recall)


# %%
# Another standard (built-in) example
# this classifier target has 3 possible outcomes
# Ref: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine
import sklearn.datasets
wine = sklearn.datasets.load_wine()
print(wine)



# %%
# Fit Logit
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(wine.data, wine.target)
lr.score(wine.data, wine.target) # accuracy score


# %%
# We can also get the probability for each row, or being class0, class1, or class2
lr.predict_proba(wine.data[:10]) # predicted 


# %%
# Performance metrics
y_true, y_pred = wine.target.copy(), lr.predict(wine.data)
print(classification_report(y_true, y_pred))



# %%
# K-Nearest-Neighbor KNN on admissions data
from sklearn.neighbors import KNeighborsClassifier

# %%
# 2-KNN algorithm
K = 2
knn_split = KNeighborsClassifier(n_neighbors= K) # instantiate with K value given
knn_split.fit(x_trainAdmit, y_trainAdmit)
# ytest_pred = knn_split.predict(x_testAdmit)
# ytest_pred
print(knn_split.score(x_testAdmit,y_testAdmit))



# %%
# Try different K values
for K in range(1, 20):
  knn_split = KNeighborsClassifier(n_neighbors= K) # instantiate with K value given
  knn_split.fit(x_trainAdmit, y_trainAdmit)
  print(K, knn_split.score(x_testAdmit,y_testAdmit))



# %%
from sklearn.model_selection import cross_val_score

K = 3 # For K-NN
k = 5 # For k-fold cross-validation
knn_cv = KNeighborsClassifier(n_neighbors = K) # instantiate with K value given

cv_results = cross_val_score(knn_cv, x_trainAdmit, y_trainAdmit, cv = k) 
print(cv_results) 
print(np.mean(cv_results)) 


# %%
# Try CV for different K
for K in range(1, 20):
  knn_cv = KNeighborsClassifier(n_neighbors = K)
  cv_results = cross_val_score(knn_cv, x_trainAdmit, y_trainAdmit, cv = k) 
  print(K, np.mean(cv_results)) 


# %%
# Scale first? better or not?
# Re-do our analysis with scale on X
from sklearn.preprocessing import scale
xsadmit = pd.DataFrame( scale(xadmit), columns=xadmit.columns )  # reminder: xadmit = dfadmit[['gre', 'gpa', 'rank']]
# Note that scale( ) coerce the object from pd.dataframe to np.array  
# Need to reconstruct the pandas df with column names
# xsadmit.rank = xadmit.rank
ysadmit = yadmit.copy()  # no need to scale y, but make a true copy / deep copy to be safe



# %%
# from sklearn.neighbors import KNeighborsClassifier
knn_scv = KNeighborsClassifier(n_neighbors = K) # instantiate with n value given

# from sklearn.model_selection import cross_val_score
scv_results = cross_val_score(knn_scv, xsadmit, ysadmit, cv=5)
print(scv_results) 
print(np.mean(scv_results)) 


#%%
# Your turn. 
# (1) Predict survival of the titanic passenger?
# (2) Predict the pclass of the titanic passenger?
# (3) Use the wine dataset to classify the target class ['class_0', 'class_1', 'class_2']
# Try both unscaled and scaled data, 
# Only need to do the CV method for in all cases. 
# (4) Now we have logit and KNN models to predict admit in the first dataset, and 
# the class for the wine dataset, compare the accuracies of these results 
# with the logit regression results.




# %%
# K-means 
# 
from sklearn.cluster import KMeans

km_xpizza = KMeans( n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0 )
y_km = km_xpizza.fit_predict(xpizza)

# %%
# plot
# plot the 3 clusters
index1 = 2
index2 = 5

plt.scatter( xpizza[y_km==0].iloc[:,index1], xpizza[y_km==0].iloc[:,index2], s=50, c='lightgreen', marker='s', edgecolor='black', label='cluster 1' )

plt.scatter( xpizza[y_km==1].iloc[:,index1], xpizza[y_km==1].iloc[:,index2], s=50, c='orange', marker='o', edgecolor='black', label='cluster 2' )

plt.scatter( xpizza[y_km==2].iloc[:,index1], xpizza[y_km==2].iloc[:,index2], s=50, c='lightblue', marker='v', edgecolor='black', label='cluster 3' )

# plot the centroids
plt.scatter( km_xpizza.cluster_centers_[:, index1], km_xpizza.cluster_centers_[:, index2], s=250, marker='*', c='red', edgecolor='black', label='centroids' )
plt.legend(scatterpoints=1)
plt.xlabel(str(index1) + " : " + xpizza.columns[index1])
plt.ylabel(str(index2) + " : " + xpizza.columns[index2])
plt.grid()
plt.show()





# %%
# Wine dataset, KNN
#
import sklearn
from sklearn.preprocessing import scale

wine = sklearn.datasets.load_wine()
for i in (3,5,7):
  knnn = i
  x_wine = wine.data
  y_wine = wine.target
  x_wine_s = pd.DataFrame( scale(x_wine) )
  y_wine_s = y_wine.copy()
  knn_wine = KNeighborsClassifier(n_neighbors=knnn)
  wine_result = cross_val_score(knn_wine, x_wine, y_wine, cv=5)
  wine_result_s = cross_val_score(knn_wine, x_wine_s, y_wine_s, cv=5)
  knn_wine.fit(x_wine, y_wine)
  a = knn_wine.score(x_wine, y_wine)  
  knn_wine.fit(x_wine_s, y_wine_s)
  b = knn_wine.score(x_wine_s, y_wine_s)
  print('knn=', knnn)
  print('method 1 - cross_val_score():')
  print('unscale:', round(np.mean(wine_result),4))
  print('scale:', round(np.mean(wine_result_s),4))
  print()
  print('method 2 - knn_wine.score():')
  print('unscale:', round(a,4))
  print('scale:', round(b,4))