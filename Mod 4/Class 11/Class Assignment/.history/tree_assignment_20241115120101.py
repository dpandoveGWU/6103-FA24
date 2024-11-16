#Import OJ data set which is part of the ISLP  package. 

#(a) Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.  
# (b) Fit a tree to the training data, with Purchase as the response  and the other variables as predictors. What is the training error  rate?  
# (c) Create a plot of the tree, and interpret the results. How many  terminal nodes does the tree have?  
# (d) Use the export_tree() function to produce a text summary of  the ftted tree. Pick one of the terminal nodes, and interpret the  information displayed.  
# (e) Predict the response on the test data, and produce a confusion  matrix comparing the test labels to the predicted test labels.  What is the test error rate? 

# %%
import statsmodels.api as sm

# Load dataset if available
oj_data = sm.datasets.get_rdataset("OJ", "ISLR").data
print(oj_data.head())

# %%
# Convert categorical target variable 'Purchase' to numeric
oj_data['Purchase'] = oj_data['Purchase'].astype('category').cat.codes

# Separate predictors (X) and the response variable (y)
X = oj_data.drop(columns=['Purchase'])
y = oj_data['Purchase']

# %%
from sklearn.model_selection import train_test_split

# Split into training (800 samples) and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, random_state=42)
