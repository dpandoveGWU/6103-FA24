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
from sklearn.tree import DecisionTreeClassifier

# Initialize and fit a decision tree model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Predict on the training set
y_train_pred = tree_model.predict(X_train)

# Calculate the training error rate
training_error_rate = 1 - (y_train_pred == y_train).mean()
print(f"Training Error Rate: {training_error_rate:.4f}")
# %%
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=X.columns, class_names=['CH', 'MM'], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# Get the number of terminal nodes
n_terminal_nodes = tree_model.get_n_leaves()
print(f"Number of Terminal Nodes: {n_terminal_nodes}")

# %%
