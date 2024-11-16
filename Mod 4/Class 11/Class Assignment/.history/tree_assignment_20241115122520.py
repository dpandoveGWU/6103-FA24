#Import OJ data set which is part of the ISLP  package. 

#(a) Create a training set containing a random sample of 800 observations, and a test set containing the remaining observations.  
# (b) Fit a tree to the training data, with Purchase as the response  and the other variables as predictors. What is the training error  rate?  
# (c) Create a plot of the tree, and interpret the results. How many  terminal nodes does the tree have?  
# (d) Use the export_tree() function to produce a text summary of  the ftted tree. Pick one of the terminal nodes, and interpret the  information displayed.  
# (e) Predict the response on the test data, and produce a confusion  matrix comparing the test labels to the predicted test labels.  What is the test error rate? 

# %%
import statsmodels.api as sm

# Load dataset
oj_data = sm.datasets.get_rdataset("OJ", "ISLR").data
print(oj_data.head())

# %%


# %%
# Convert categorical target variable 'Purchase' to numeric
oj_data['Purchase'] = oj_data['Purchase'].astype('category').cat.codes

# Separate predictors (X) and the response variable (y)
X = oj_data.drop(columns=['Purchase'])
y = oj_data['Purchase']
# %%
# Check data types of all columns
print(oj_data.dtypes)

# Identify non-numeric columns
non_numeric_columns = oj_data.select_dtypes(include=['object', 'category']).columns
print("Non-numeric columns:", non_numeric_columns)

# Manually encode 'Store7' column
oj_data['Store7'] = oj_data['Store7'].apply(lambda x: 1 if x == 'Yes' else 0)

# Verify encoding
print(oj_data['Store7'].unique())

# %%
# Check unique values in 'Store7'
print(oj_data['Store7'].unique())

# Check all columns for non-numeric types or unexpected data
print(oj_data.info())
# %%
# Replace 'Yes' with 1 and 'No' with 0
oj_data['Store7'] = oj_data['Store7'].replace({'Yes': 1, 'No': 0})

# Confirm conversion
print(oj_data['Store7'].unique())

# %%
# Check if all columns are numeric
print(oj_data.dtypes)

# Ensure no unexpected string data remains
print(oj_data.head())
# %%
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Separate predictors (X) and response variable (y)
X = oj_data.drop(columns=['Purchase'])
y = oj_data['Purchase']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, random_state=42)

# Fit the decision tree
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Calculate training error rate
y_train_pred = tree_model.predict(X_train)
training_error_rate = 1 - (y_train_pred == y_train).mean()
print(f"Training Error Rate: {training_error_rate:.4f}")


# %%


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
from sklearn.tree import export_text

# Export the text representation of the tree
tree_summary = export_text(tree_model, feature_names=list(X.columns))
print(tree_summary)

# Interpret one terminal node from the summary manually.


# %%
from sklearn.metrics import confusion_matrix, classification_report

# Predict on the test set
y_test_pred = tree_model.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Test error rate
test_error_rate = 1 - (y_test_pred == y_test).mean()
print(f"Test Error Rate: {test_error_rate:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))

