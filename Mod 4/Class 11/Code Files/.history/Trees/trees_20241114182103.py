# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# %%
# Load data 
dfadmit = pd.read_csv('gradAdmit.csv')


# %%
# Prepare data for the model
xadmit = dfadmit[['gre', 'gpa', 'rank']]
yadmit = dfadmit['admit']

# Split the data for training
X_train, X_test, y_train, y_test= train_test_split(xadmit, yadmit, 
                                                   test_size = 0.2, random_state = 1)

row(y_train)
# %%
# Decision Trees for Classification
# y-target is categorical, similar to KNN, (multinomial) logistic Regression
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier


# %%
# Activity: 1
# Identify some pros and cons of decision trees
#How are they different from Regression models.


# %%
# Instantiate dtree
dtree_admit1 = DecisionTreeClassifier(max_depth = 3, random_state = 1)
#This classifier will be used to predict or classify data by learning simple decision rules from the features.
#By setting parameters like max_depth, we can control the tree’s complexity.

# Fit dt to the training set
dtree_admit1.fit(X_train, y_train)

# Predict test set labels
y_test_pred = dtree_admit1.predict(X_test)
#In this example, the model dtree_admit1 is trained on X_train (features) and y_train (target values). Then, it can be used to predict the class labels of new data (like X_test).

# Evaluate test-set accuracy
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

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



# %% [markdown]
#
# Considering the original dataset has 273 rejections and 127 acceptances out of 400 (68.25% and 31.75%) 
# respectively, the model accuracy here is really terrible. 
# From the scatterplot, one most likely will see admitted/rejected are scattered very close to each other 
# and have very little pattern to work with. 

# %% 
# Let us try different citeria...
# Instantiate dtree, try criterion='gini' (default)  or 'entropy'
maxlevel = None # default is None
crit = 'gini' # 'gini' default, other option: 'entropy'
#The Gini impurity measures how often a randomly chosen element from the set would be incorrectly classified. 
# The other option, 'entropy', uses information gain for splitting.

dtree_admit2 = DecisionTreeClassifier(max_depth = maxlevel, criterion = crit, random_state = 1)
# Fit dt to the training set
dtree_admit2.fit(X_train, y_train)
#This trains (fits) the decision tree model on the training data, X_train (features) and y_train (labels/targets).
# Predict test set labels
y_train_pred = dtree_admit2.predict(X_train)
y_test_pred = dtree_admit2.predict(X_test)
print(f'max_level: {maxlevel} ; criterion: {crit} ')

# Evaluate train-set accuracy
print('train set evaluation: ')
print(accuracy_score(y_train, y_train_pred))
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

# Evaluate test-set accuracy
print('test set evaluation: ')
print(accuracy_score(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))



# %%
# Activity: 2
# Find the best max-level and criteria 
# Let us loop through the codes above
maxlevels = [None, 2, 3, 5, 8]
crits = ['gini', 'entropy']
for l in maxlevels:
    for c in crits:
        dt = DecisionTreeClassifier(max_depth = l, criterion = c)
        dt.fit(X_train, y_train)
        print(l, c, dt.score(X_test, y_test))

# Looks like `maxlevel = 5` and `crit = entropy` is the best

# %% 
# compare with logistic regression result
logitreg_admit = LogisticRegression( )

# Fit logitreg_admit to the training set
logitreg_admit_fit = logitreg_admit.fit(X_train, y_train)
print(logitreg_admit_fit.score(X_test, y_test))



# %%
# Use function plot_decision_regions from mlxtend.plotting
# %pip install mlxtend 
# %pip3 install mlxtend
# %conda install mlxtend
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


# %%
# Now plot
# Review the decision regions of the two classifiers

# Plotting decision regions
# plot_decision_regions(X_test.values, y_test.values, clf=logitreg_admit)
plot_decision_regions(X_test.values, y_test.values, clf=logitreg_admit, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2:3} )
# filler_feature_values is used when you have more than 2 predictors, then 
# you need to specify the ones not shown in the 2-D plot. 
# For example, assume we have three predictors: gre, gpa, and rank.
# For us, the rank is at position 2, and the value can be 1, 2, 3, or 4.
# also need to specify the filler_feature_ranges for +/-, otherwise only data points with that feature value will be shown.
#
# Adding axes annotations
plt.xlabel(X_test.columns[0])
plt.ylabel(X_test.columns[1])
plt.title(logitreg_admit.__class__.__name__)
plt.show()

# %%
# And the decision tree result
plot_decision_regions(X_test.values, y_test.values, clf=dtree_admit1, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2:3} )
plt.xlabel(X_test.columns[0])
plt.ylabel(X_test.columns[1])
plt.title(dtree_admit1.__class__.__name__)
plt.show()



# %% [markdown]
# Notice that decision trees only make orthogonal cuts. 
#Activity
# Try to change the depth of the tree, and see what happens. You can also try the gini and entropy criteria
plot_decision_regions(X_test.values, y_test.values, clf=dtree_admit1, legend=3, filler_feature_values={2:1} , filler_feature_ranges={2: 3} )
plt.xlabel(X_test.columns[0])
plt.ylabel(X_test.columns[1])
plt.title(dtree_admit1.__class__.__name__)
plt.show()


# %%
# Review pizza dataset
dfpizza = pd.read_csv('Pizza.csv')


# %%
# Prepare our X data (features, predictors, regressors) and y data (target, dependent variable)
xpizza = dfpizza[['prot', 'fat', 'sodium']]
ypizza = dfpizza['cal']




# %%
# Regression Trees (y is numerical variable)
from sklearn.tree import DecisionTreeRegressor  # Import DecisionTreeRegressor
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import mean_squared_error as MSE  # Import mean_squared_error as MSE
# Split data into 80% train and 20% test
X_train, X_test, y_train, y_test= train_test_split(xpizza, ypizza, test_size=0.2,random_state=1)
# Instantiate a DecisionTreeRegressor 'regtree0'
regtree0 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.1,random_state=22) 
# set minimum leaf to contain at least 10% of data points
# DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
#     max_leaf_nodes=None, min_impurity_decrease=0.0,
#     min_impurity_split=None, min_samples_leaf=0.13,
#     min_samples_split=2, min_weight_fraction_leaf=0.0,
#     presort=False, random_state=3, splitter='best')


regtree0.fit(X_train, y_train)  # Fit regtree0 to the training set
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE

# evaluation
y_test_pred = regtree0.predict(X_test)  # Compute y_test_pred
mse_regtree0 = MSE(y_test, y_test_pred)  # Compute mse_regtree0
rmse_regtree0 = mse_regtree0 ** (.5) # Compute rmse_regtree0
print("Test set RMSE of regtree0: {:.2f}".format(rmse_regtree0))


# %%
# Let us compare the performance with OLS
from sklearn import linear_model
olspizza = linear_model.LinearRegression() 
olspizza.fit( X_train, y_train )

y_pred_ols = olspizza.predict(X_test)  # Predict test set labels/values

mse_ols = MSE(y_test, y_pred_ols)  # Compute mse_ols
rmse_ols = mse_ols**(0.5)  # Compute rmse_ols

print('Linear Regression test set RMSE: {:.2f}'.format(rmse_ols))
print('Regression Tree test set RMSE: {:.2f}'.format(rmse_regtree0))

# You can try to include different ingredients, and see how the two compare. 
# If you include both 'carb' and 'fat', the OLS model is almost perfect. Reg Tree cannot compete there. 
# In most other combinations, Reg Tree is measured up against OLS pretty good.


#%% [markdown]
#
# #  Bias-variance tradeoff  
# high bias: underfitting  
# high variance: overfitting, too much complexity  
# Generalization Error = (bias)^2 + Variance + irreducible error  
# 
# Solution: Use CV  
# 
# 1. If CV error (average of 10- or 5-fold) > training set error  
#   - high variance
#   - overfitting the training set
#   - try to decrease model complexity
#   - decrease max depth
#   - increase min samples per leaf
#   - get more data
# 2. If CV error approximates the training set error, and greater than desired error
#   - high bias
#   - underfitting the training set
#   - increase max depth
#   - decrease min samples per leaf
#   - use or gather more relevant features


# %%
# We already have train-test split at 75-25%, let us compare the result with 10-fold CV average
# Split the data into 70% train and 30% test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
SEED = 28
regtree1 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=0.22, random_state=SEED)

# Evaluate the list of MSE ontained by 10-fold CV
from sklearn.model_selection import cross_val_score
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(regtree1, X_train, y_train, cv= 10, scoring='neg_mean_squared_error')
regtree1.fit(X_train, y_train)  # Fit 'regtree1' to the training set
y_predict_train = regtree1.predict(X_train)  # Predict the labels of training set
y_predict_test = regtree1.predict(X_test)  # Predict the labels of test set

print('CV RMSE:', MSE_CV.mean()**(0.5) )  #CV MSE 
print('Training set RMSE:', MSE(y_train, y_predict_train)**(0.5) )   # Training set MSE
print('Test set RMSE:', MSE(y_test, y_predict_test)**(0.5) )   # Test set MSE 
print("\nReady to continue.")

#%%
# Is it high bias? high variance? 
# If so, change some parameters and try?

# Other topics:
# bagging
# boosting
# tuning, pruning

print("\nReady to continue.")

#%%
# Graphing the tree
from sklearn.tree import export_graphviz  

# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 

filename = 'tree1'
# import os
# print(os.getcwd())
export_graphviz(dtree_admit1, out_file = filename + '.dot' , feature_names =['gre', 'gpa']) 

# can't get it to work yet
# pip3 install pydot
# https://graphviz.gitlab.io/download/
# for MacOS, I ran into error, 
# No such file or directory: 'dot': 'dot' (on graph.write_png)
# Need to install graphviz NOT using pip 
# %brew install graphviz 
# or sudo brew install graphviz 
# but 
# before that, you might need to install homebrew
# %ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
# but
# before that, make sure your XCode is install and works. 
# My XCode needs to be removed and reinstalled because of the latest MacOS upgrade.
# Even if your Xcode is fine, it most like will still download and install 
# XCode_cli (command-line-interface, not installed by default from Mac), and 
# that takes quite a while. 
# 
# import pydot
# (graph,) = pydot.graph_from_dot_file(filename+'.dot')
# graph.write_png(filename+'.png')


# %%
