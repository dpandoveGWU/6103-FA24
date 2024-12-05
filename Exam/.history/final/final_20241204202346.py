# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional

world1 = pd.read_csv("world1.csv", index_col="id")
world2 = pd.read_csv("world2.csv", index_col="id") 

print("\nReady to continue.")

#%% [markdown]
# # Two Worlds (Continuation from midterm: Part I - 25%)
# 
# In the (midterm) mini-project, we used statistical tests and visualization to 
# studied these two worlds. Now let us use the modeling techniques we now know
# to give it another try. 
# 
# Use appropriate models that we learned in this class or elsewhere, 
# elucidate what these two world looks like. 
# 
# Having an accurate model (or not) however does not tell us if the worlds are 
# utopia or not. Is it possible to connect these concepts together? (Try something called 
# "feature importance"?)
# 
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import altair as alt
import numpy as np




# %%
# Reading the datasets
world1 = pd.read_csv("world1.csv")
world2 = pd.read_csv("world2.csv")

# Checking for missing values
world1_missing = world1.isnull().sum()
world2_missing = world2.isnull().sum()

# Display structure and summaries to user
world1_missing, world2_missing

# %%

# %%
# Linear regression for World1
# Features and target
X = world1.drop(columns=["income00"])
y = world1["income00"]

# Identify categorical and numerical columns
categorical_columns = ["marital", "gender", "ethnic", "industry"]
numerical_columns = ["age00", "education"]

# Encoding categorical variables and scaling numerical ones
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(drop="first"), categorical_columns)
    ]
)

# Apply transformations
X_preprocessed = preprocessor.fit_transform(X)

# Retrieve feature names for mapping
numerical_features = numerical_columns
categorical_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
all_features = ["const"] + list(np.concatenate([numerical_features, categorical_features]))

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

# Ensure indices of X_train match y_train
X_train_sm = pd.DataFrame(sm.add_constant(X_train), columns=all_features, index=y_train.index)

# Fit the Linear Regression model using statsmodels for World1
linear_model_sm_world1 = sm.OLS(y_train, X_train_sm).fit()
print("World1 Linear Regression Summary with Feature Names:")
print(linear_model_sm_world1.summary())

# Predict and evaluate World1
y_pred = linear_model_sm_world1.predict(pd.DataFrame(sm.add_constant(X_test), columns=all_features))
rmse_world1 = root_mean_squared_error(y_test, y_pred)
r2_world1 = r2_score(y_test, y_pred)

# Linear regression for World2
X_world2 = world2.drop(columns=["income00"])
y_world2 = world2["income00"]

# Preprocessing for World2
X_world2_preprocessed = preprocessor.transform(X_world2)

# Split World2 data into training and testing sets
X_train_world2, X_test_world2, y_train_world2, y_test_world2 = train_test_split(
    X_world2_preprocessed, y_world2, test_size=0.3, random_state=42
)

# Ensure indices of X_train_world2 match y_train_world2
X_train_world2_sm = pd.DataFrame(sm.add_constant(X_train_world2), columns=all_features, index=y_train_world2.index)

# Fit the Linear Regression model using statsmodels for World2
linear_model_sm_world2 = sm.OLS(y_train_world2, X_train_world2_sm).fit()
print("\nWorld2 Linear Regression Summary with Feature Names:")
print(linear_model_sm_world2.summary())

# Predict and evaluate World2
y_pred_world2 = linear_model_sm_world2.predict(pd.DataFrame(sm.add_constant(X_test_world2), columns=all_features))
rmse_world2 = root_mean_squared_error(y_test_world2, y_pred_world2)
r2_world2 = r2_score(y_test_world2, y_pred_world2)

# Print final evaluation metrics for both worlds
print("\nFinal Performance Metrics:")
print("World1:")
print(f"Root Mean Squared Error (RMSE): {rmse_world1:.2f}")
print(f"R^2 Score: {r2_world1:.4f}")

print("\nWorld2:")
print(f"Root Mean Squared Error (RMSE): {rmse_world2:.2f}")
print(f"R^2 Score: {r2_world2:.4f}")


# %%[markdown]
# For world1 The linear regression model explains 85.01% of the variability in income using features such as age, education, gender, marital status, and industry.
# For world2 The model explains 84.50% of the variability in income using the same features.
# In both World1 and World2, industries are the most significant predictors of income, with higher-paying industries (e.g., finance, professional services) showing the largest impacts since their p-value is small.
# Age, education, gender, and ethnicity are not statistically significant predictors of income in either world (p-values > 0.05).
# %%
# Retrieve feature names from the preprocessor
numerical_features = numerical_columns
categorical_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_columns)
all_features = ["const"] + list(np.concatenate([numerical_features, categorical_features]))

# Feature importance for World1
coefficients_world1 = linear_model_sm_world1.params  # Access coefficients using `params`
feature_importance_world1 = pd.DataFrame({
    "Feature": all_features,
    "Importance": np.abs(coefficients_world1)
}).sort_values(by="Importance", ascending=False)

# Feature importance for World2
coefficients_world2 = linear_model_sm_world2.params  # Access coefficients using `params`
feature_importance_world2 = pd.DataFrame({
    "Feature": all_features,
    "Importance": np.abs(coefficients_world2)
}).sort_values(by="Importance", ascending=False)

# Bar chart for World1
world1_chart = alt.Chart(feature_importance_world1).mark_bar().encode(
    x=alt.X("Importance", title="Feature Importance"),
    y=alt.Y("Feature", sort="-x", title="Feature"),
    tooltip=["Feature", "Importance"]
).properties(
    title="Feature Importance - World1 (Linear Regression)",
    width=600,
    height=400
)

# Bar chart for World2
world2_chart = alt.Chart(feature_importance_world2).mark_bar().encode(
    x=alt.X("Importance", title="Feature Importance"),
    y=alt.Y("Feature", sort="-x", title="Feature"),
    tooltip=["Feature", "Importance"]
).properties(
    title="Feature Importance - World2 (Linear Regression)",
    width=600,
    height=400
)

# Display the charts
world1_chart.display()
world2_chart.display()

# Display top 10 features for both worlds
print("Top 10 Feature Importance - World1 (Linear Regression):")
print(feature_importance_world1.head(10))

print("\nTop 10 Feature Importance - World2 (Linear Regression):")
print(feature_importance_world2.head(10))

# Save the full feature importance to CSV for both datasets
feature_importance_world1.to_csv("feature_importance_world1.csv", index=False)
feature_importance_world2.to_csv("feature_importance_world2.csv", index=False)

print("\nFeature importance saved as 'feature_importance_world1.csv' and 'feature_importance_world2.csv'.")


# %%[markdown]
# The industries are the most important features for determining income in both World1 and World2, with `industry_7` (finance) and `industry_6` (professional/business) having the highest influence. 
# Other features, such as marital status, ethnicity, age, education, and gender, are less significant in both worlds as they have higher p-value.

# %%



# %%
# with Interaction terms
# logistic regression models for World1 and World2

#Prepare Binary Target
median_income_world1 = world1['income00'].median()
world1['target'] = (world1['income00'] > median_income_world1).astype(int)

median_income_world2 = world2['income00'].median()
world2['target'] = (world2['income00'] > median_income_world2).astype(int)

# Helper function to evaluate and display results
def train_logistic_model(X_train, y_train, X_test, y_test, title):
    X_train_sm = sm.add_constant(X_train)  # Add constant for intercept
    logit_model = sm.Logit(y_train, X_train_sm).fit()

    # Predict probabilities and classes
    X_test_sm = sm.add_constant(X_test)
    y_prob = logit_model.predict(X_test_sm)
    y_pred = (y_prob >= 0.5).astype(int)

    # Evaluate Metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.show()

    # Print Outputs
    print(f"\n{title} Summary:")
    print(logit_model.summary())
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")
    return logit_model

# Incremental Model Building for World1
# Step 1: Model with Industry Only
X_world1_industry = pd.get_dummies(world1[['industry']], drop_first=True)
X_train_w1, X_test_w1, y_train_w1, y_test_w1 = train_test_split(X_world1_industry, world1['target'], test_size=0.3, random_state=42)
print("\nWorld1 Base Model (Industry Only):")
logit_model_w1 = train_logistic_model(X_train_w1, y_train_w1, X_test_w1, y_test_w1, "World1 - Industry Only")

# Step 2: Model with Industry + Education
X_world1_edu = pd.concat([X_world1_industry, world1[['education']]], axis=1)
X_train_w1, X_test_w1, y_train_w1, y_test_w1 = train_test_split(X_world1_edu, world1['target'], test_size=0.3, random_state=42)
print("\nWorld1 Model (Industry + Education):")
logit_model_w1_edu = train_logistic_model(X_train_w1, y_train_w1, X_test_w1, y_test_w1, "World1 - Industry + Education")

# Step 3: Model with Industry + Education + Age
X_world1_age = pd.concat([X_world1_edu, world1[['age00']]], axis=1)
X_train_w1, X_test_w1, y_train_w1, y_test_w1 = train_test_split(X_world1_age, world1['target'], test_size=0.3, random_state=42)
print("\nWorld1 Model (Industry + Education + Age):")
logit_model_w1_age = train_logistic_model(X_train_w1, y_train_w1, X_test_w1, y_test_w1, "World1 - Industry + Education + Age")

# Repeat the same steps for World2
# Step 1: Model with Industry Only
X_world2_industry = pd.get_dummies(world2[['industry']], drop_first=True)
X_train_w2, X_test_w2, y_train_w2, y_test_w2 = train_test_split(X_world2_industry, world2['target'], test_size=0.3, random_state=42)
print("\nWorld2 Base Model (Industry Only):")
logit_model_w2 = train_logistic_model(X_train_w2, y_train_w2, X_test_w2, y_test_w2, "World2 - Industry Only")

# Step 2: Model with Industry + Education
X_world2_edu = pd.concat([X_world2_industry, world2[['education']]], axis=1)
X_train_w2, X_test_w2, y_train_w2, y_test_w2 = train_test_split(X_world2_edu, world2['target'], test_size=0.3, random_state=42)
print("\nWorld2 Model (Industry + Education):")
logit_model_w2_edu = train_logistic_model(X_train_w2, y_train_w2, X_test_w2, y_test_w2, "World2 - Industry + Education")

# Step 3: Model with Industry + Education + Age
X_world2_age = pd.concat([X_world2_edu, world2[['age00']]], axis=1)
X_train_w2, X_test_w2, y_train_w2, y_test_w2 = train_test_split(X_world2_age, world2['target'], test_size=0.3, random_state=42)
print("\nWorld2 Model (Industry + Education + Age):")
logit_model_w2_age = train_logistic_model(X_train_w2, y_train_w2, X_test_w2, y_test_w2, "World2 - Industry + Education + Age")

# %%[markdown]
# The Logistic regression model with only Industry as a predictor performed well with an accurracy of 89%.
# Adding other predictors didn't improve the model performance, this indicates that other predictors have less significant impact on income.

# %%
# Decision Tree models
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load World1 and World2 datasets (already provided in your workspace)

# Data dictionary
categorical_columns = ["marital", "gender", "ethnic", "industry"]
numerical_columns = ["age00", "education"]

# Preprocessing: Scaling numerical and encoding categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(drop="first"), categorical_columns)
    ]
)

# Apply preprocessing to World1
X_world1 = world1.drop(columns=["income00", "target"], errors="ignore")  # Exclude target variables
y_world1_reg = world1["income00"]  # Target for regression
y_world1_class = world1["target"]  # Target for classification (binary target)
X_world1_preprocessed = preprocessor.fit_transform(X_world1)

# Apply preprocessing to World2
X_world2 = world2.drop(columns=["income00", "target"], errors="ignore")
y_world2_reg = world2["income00"]
y_world2_class = world2["target"]
X_world2_preprocessed = preprocessor.transform(X_world2)

# Split datasets into training and testing sets
# World1
X_train_reg_w1, X_test_reg_w1, y_train_reg_w1, y_test_reg_w1 = train_test_split(
    X_world1_preprocessed, y_world1_reg, test_size=0.3, random_state=42
)
X_train_class_w1, X_test_class_w1, y_train_class_w1, y_test_class_w1 = train_test_split(
    X_world1_preprocessed, y_world1_class, test_size=0.3, random_state=42
)

# World2
X_train_reg_w2, X_test_reg_w2, y_train_reg_w2, y_test_reg_w2 = train_test_split(
    X_world2_preprocessed, y_world2_reg, test_size=0.3, random_state=42
)
X_train_class_w2, X_test_class_w2, y_train_class_w2, y_test_class_w2 = train_test_split(
    X_world2_preprocessed, y_world2_class, test_size=0.3, random_state=42
)



# Helper function to evaluate regression models
def evaluate_regression_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    deviance = np.var(y_test - y_pred)

    # Print Outputs
    print(f"\n{title} Performance:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Deviance: {deviance:.2f}")
    return y_pred

# Helper function to evaluate classification models
def evaluate_classification_model(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    # Metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.show()

    # Print Outputs
    print(f"\n{title} Performance:")
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

# 1. Decision Tree Regression for World1
print("\nDecision Tree Regression - World1:")
tree_reg_w1 = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg_w1.fit(X_train_reg_w1, y_train_reg_w1)
evaluate_regression_model(tree_reg_w1, X_test_reg_w1, y_test_reg_w1, "Decision Tree Regression - World1")

# Add cost complexity pruning for World1 regression
print("\nDecision Tree Regression with Pruning - World1:")
path_reg_w1 = tree_reg_w1.cost_complexity_pruning_path(X_train_reg_w1, y_train_reg_w1)
ccp_alphas_reg_w1 = path_reg_w1.ccp_alphas
for alpha in ccp_alphas_reg_w1[::3]:  # Test pruning with every 3rd alpha for efficiency
    tree_reg_pruned_w1 = DecisionTreeRegressor(max_depth=5, random_state=42, ccp_alpha=alpha)
    tree_reg_pruned_w1.fit(X_train_reg_w1, y_train_reg_w1)
    print(f"Alpha: {alpha}")
    evaluate_regression_model(tree_reg_pruned_w1, X_test_reg_w1, y_test_reg_w1, f"Decision Tree Regression Pruned (α={alpha}) - World1")

# 2. Classification Tree for World1
print("\nClassification Tree - World1:")
tree_class_w1 = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_class_w1.fit(X_train_class_w1, y_train_class_w1)
evaluate_classification_model(tree_class_w1, X_test_class_w1, y_test_class_w1, "Classification Tree - World1")

# Add cost complexity pruning for World1 classification
print("\nClassification Tree with Pruning - World1:")
path_class_w1 = tree_class_w1.cost_complexity_pruning_path(X_train_class_w1, y_train_class_w1)
ccp_alphas_class_w1 = path_class_w1.ccp_alphas
for alpha in ccp_alphas_class_w1[::3]:  # Test pruning with every 3rd alpha for efficiency
    tree_class_pruned_w1 = DecisionTreeClassifier(max_depth=5, random_state=42, ccp_alpha=alpha)
    tree_class_pruned_w1.fit(X_train_class_w1, y_train_class_w1)
    print(f"Alpha: {alpha}")
    evaluate_classification_model(tree_class_pruned_w1, X_test_class_w1, y_test_class_w1, f"Classification Tree Pruned (α={alpha}) - World1")

# Repeat the same process for World2
print("\nDecision Tree Regression - World2:")
tree_reg_w2 = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg_w2.fit(X_train_reg_w2, y_train_reg_w2)
evaluate_regression_model(tree_reg_w2, X_test_reg_w2, y_test_reg_w2, "Decision Tree Regression - World2")

print("\nClassification Tree - World2:")
tree_class_w2 = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_class_w2.fit(X_train_class_w2, y_train_class_w2)
evaluate_classification_model(tree_class_w2, X_test_class_w2, y_test_class_w2, "Classification Tree - World2")

# %%[markdown]
# For the decision tree I performed Pruning applied using cost complexity pruning (`ccp_alpha`), progressively adjusting alpha values to balance model complexity and performance.
# and it improved the model performance until \( \alpha = 162,928.86 \), beyond which over-pruning reduced accuracy.
# The unpruned classification tree had better overall accuracy (83%) and ROC-AUC (0.91), while pruning improved recall (97%) but reduced accuracy (76%) and ROC-AUC (0.76), indicating a trade-off.

# %%

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Helper function to evaluate regression models
def evaluate_rf_regression(model, X_test, y_test, title):
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)

    # Print Outputs
    print(f"\n{title} Performance:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

# Helper function to evaluate classification models
def evaluate_rf_classification(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.show()

    # Print Outputs
    print(f"\n{title} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

# 1. Random Forest Regression - World1
print("\nRandom Forest Regression - World1:")
rf_reg_w1 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg_w1.fit(X_train_reg_w1, y_train_reg_w1)
evaluate_rf_regression(rf_reg_w1, X_test_reg_w1, y_test_reg_w1, "Random Forest Regression - World1")

# 2. Random Forest Regression - World2
print("\nRandom Forest Regression - World2:")
rf_reg_w2 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg_w2.fit(X_train_reg_w2, y_train_reg_w2)
evaluate_rf_regression(rf_reg_w2, X_test_reg_w2, y_test_reg_w2, "Random Forest Regression - World2")

# 3. Random Forest Classification - World1
print("\nRandom Forest Classification - World1:")
rf_class_w1 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_class_w1.fit(X_train_class_w1, y_train_class_w1)
evaluate_rf_classification(rf_class_w1, X_test_class_w1, y_test_class_w1, "Random Forest Classification - World1")

# 4. Random Forest Classification - World2
print("\nRandom Forest Classification - World2:")
rf_class_w2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_class_w2.fit(X_train_class_w2, y_train_class_w2)
evaluate_rf_classification(rf_class_w2, X_test_class_w2, y_test_class_w2, "Random Forest Classification - World2")


#%% [markdown]
#
# %%
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

# Helper function to evaluate regression models
def evaluate_knn_regression(model, X_test, y_test, title):
    y_pred = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)

    # Print Outputs
    print(f"\n{title} Performance:")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R² Score: {r2:.2f}")

# Helper function to evaluate classification models
def evaluate_knn_classification(model, X_test, y_test, title):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.show()

    # Print Outputs
    print(f"\n{title} Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

# 1. KNN Regression - World1
print("\nKNN Regression - World1:")
knn_reg_w1 = KNeighborsRegressor(n_neighbors=5)  # Default k=5
knn_reg_w1.fit(X_train_reg_w1, y_train_reg_w1)
evaluate_knn_regression(knn_reg_w1, X_test_reg_w1, y_test_reg_w1, "KNN Regression - World1")

# 2. KNN Regression - World2
print("\nKNN Regression - World2:")
knn_reg_w2 = KNeighborsRegressor(n_neighbors=5)
knn_reg_w2.fit(X_train_reg_w2, y_train_reg_w2)
evaluate_knn_regression(knn_reg_w2, X_test_reg_w2, y_test_reg_w2, "KNN Regression - World2")

# 3. KNN Classification - World1
print("\nKNN Classification - World1:")
knn_class_w1 = KNeighborsClassifier(n_neighbors=5)
knn_class_w1.fit(X_train_class_w1, y_train_class_w1)
evaluate_knn_classification(knn_class_w1, X_test_class_w1, y_test_class_w1, "KNN Classification - World1")

# 4. KNN Classification - World2
print("\nKNN Classification - World2:")
knn_class_w2 = KNeighborsClassifier(n_neighbors=5)
knn_class_w2.fit(X_train_class_w2, y_train_class_w2)
evaluate_knn_classification(knn_class_w2, X_test_class_w2, y_test_class_w2, "KNN Classification - World2")

# %%[markdown]
# The KNN Classification model performed well for both World1 and World2, achieving high accuracy and balanced precision-recall with (ROC-AUC: 0.94).



# %%[markdown]
# # Free Worlds (Continuation from midterm: Part II - 25%)
# 
# To-do: Complete the method/function predictFinalIncome towards the end of this Part II codes.  
#  
# The worlds are gifted with freedom. Sort of.  
# I have a model built for them. It predicts their MONTHLY income/earning growth, 
# base on the characteristics of the individual. You task is to first examine and 
# understand the model. If you don't like it, build you own world and own model. 
# For now, please help me finish the last piece.  
# 
# My model will predict what is the growth factor for each person in the immediate month ahead. 
# Along the same line, it also calculate what is the expected (average) salary after 1 month with 
# that growth rate. You need to help make it complete, by producing a method/function that will 
# calculate what is the salary after n months. (Method: predictFinalIncome )  
# 
# That's all. Then try this model on people like Plato, and also create some of your favorite 
# people with all sort of different demographics, and see what their growth rates / growth factors 
# are in my worlds. Use the sample codes after the class definition below.  
# 
#%%
class Person:
  """ 
  a person with properties in the utopia 
  """

  def __init__(self, personinfo):
    self.age00 = personinfo['age'] # age at creation or record. Do not change.
    self.age = personinfo['age'] # age at current time. 
    self.income00 = personinfo['income'] # income at creation or record. Do not change.
    self.income = personinfo['income'] # income at current time.
    self.education = personinfo['education']
    self.gender = personinfo['gender']
    self.marital = personinfo['marital']
    self.ethnic = personinfo['ethnic']
    self.industry = personinfo['industry']
    # self.update({'age00': self.age00, 
    #         'age': self.age,
    #         'education': self.education,
    #         'gender': self.gender,
    #         'ethnic': self.ethnic,
    #         'marital': self.marital,
    #         'industry': self.industry,
    #         'income00': self.income00,
    #         'income': self.income})
    return
  
  def update(self, updateinfo):
    for key,val in updateinfo.items():
      if key in self.__dict__ : 
        self.__dict__[key] = val
    return
        
  def __getitem__(self, item):  # this will allow both person.gender or person["gender"] to access the data
    return self.__dict__[item]

  
#%%  
class myModel:
  """
  The earning growth model for individuals in the utopia. 
  This is a simplified version of what a model could look like, at least on how to calculate predicted values.
  """

  # ######## CONSTRUCTOR  #########
  def __init__(self, bias) :
    """
    :param bias: we will use this potential bias to explore different scenarios to the functions of gender and ethnicity

    :param b_0: the intercept of the model. This is like the null model. Or the current average value. 

    :param b_age: (not really a param. it's more a function/method) if the model prediction of the target is linearly proportional to age, this would the constant coefficient. In general, this does not have to be a constant, and age does not even have to be numerical. So we will treat this b_age as a function to convert the value (numerical or not) of age into a final value to be combined with b_0 and the others 
    
    :param b_education: similar. 
    
    :param b_gender: similar
    
    :param b_marital: these categorical (coded into numeric) levels would have highly non-linear relationship, which we typically use seaparate constants to capture their effects. But they are all recorded in this one function b_martial
    
    :param b_ethnic: similar
    
    :param b_industry: similar
    
    :param b_income: similar. Does higher salary have higher income or lower income growth rate as lower salary earners?
    """

    self.bias = bias # bias is a dictionary with info to set bias on the gender function and the ethnic function

    # ##################################################
    # The inner workings of the model below:           #
    # ##################################################

    self.b_0 = 0.0023 # 0.23% MONTHLY grwoth rate as the baseline. We will add/subtract from here

    # Technically, this is the end of the constructor. Don't change the indent

  # The rest of the "coefficients" b_1, b_2, etc are now disguised as functions/methods
  def b_age(self, age): # a small negative effect on monthly growth rate before age 45, and slight positive after 45
    effect = -0.00035 if (age<40) else 0.00035 if (age>50) else 0.00007*(age-45)
    return effect

  def b_education(self, education): 
    effect = -0.0006 if (education < 8) else -0.00025 if (education <13) else 0.00018 if (education <17) else 0.00045 if (education < 20) else 0.0009
    return effect

  def b_gender(self, gender):
    effect = 0
    biasfactor = 1 if ( self.bias["gender"]==True or self.bias["gender"] > 0) else 0 if ( self.bias["gender"]==False or self.bias["gender"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.00045 if (gender<1) else 0.00045  # This amount to about 1% difference annually
    return biasfactor * effect 

  def b_marital(self, marital): 
    effect = 0 # let's assume martial status does not affect income growth rate 
    return effect

  def b_ethnic(self, ethnic):
    effect = 0
    biasfactor = 1 if ( self.bias["ethnic"]==True or self.bias["ethnic"] > 0) else 0 if ( self.bias["ethnic"]==False or self.bias["ethnic"] ==0 ) else -1  # for bias, no-bias, and reverse bias
    effect = -0.0006 if (ethnic < 1) else -0.00027 if (ethnic < 2) else 0.00045 
    return biasfactor * effect

  def b_industry(self, industry):
    effect = 0 if (industry < 2) else 0.00018 if (industry <4) else 0.00045 if (industry <5) else 0.00027 if (industry < 6) else 0.00045 if (industry < 7) else 0.00055
    return effect

  def b_income(self, income):
    # This is the kicker! 
    # More disposable income allow people to invest (stocks, real estate, bitcoin). Average gives them 6-10% annual return. 
    # Let us be conservative, and give them 0.6% return annually on their total income. So say roughly 0.0005 each month.
    # You can turn off this effect and compare the difference if you like. Comment in-or-out the next two lines to do that. 
    # effect = 0
    effect = 0 if (income < 50000) else 0.0001 if (income <65000) else 0.00018 if (income <90000) else 0.00035 if (income < 120000) else 0.00045 
    # Notice that this is his/her income affecting his/her future income. It's exponential in natural. 
    return effect

    # ##################################################
    # end of black box / inner structure of the model  #
    # ##################################################

  # other methods/functions
  def predictGrowthFactor( self, person ): # this is the MONTHLY growth FACTOR
    factor = 1 + self.b_0 + self.b_age( person["age"] ) + self.b_education( person['education'] ) + self.b_ethnic( person['ethnic'] ) + self.b_gender( person['gender'] ) + self.b_income( person['income'] ) + self.b_industry( person['industry'] ) + self.b_marital( ['marital'] )
    # becareful that age00 and income00 are the values of the initial record of the dataset/dataframe. 
    # After some time, these two values might have changed. We should use the current values 
    # for age and income in these calculations.
    return factor

  def predictIncome( self, person ): # perdict the new income one MONTH later. (At least on average, each month the income grows.)
    return person['income']*self.predictGrowthFactor( person )

  
  def predictFinalIncome(self, n, person):
    """
    # predict final income after n months from the initial record.
    # the right codes should be no longer than a few lines.
    # If possible, please also consider the fact that the person is getting older by the month. 
    # The variable age value keeps changing as we progress with the future prediction.
    """
    for _ in range(n):
        person.age += 1 / 12  # Increment age by 1/12 for each month
        person.income *= self.predictGrowthFactor(person)  # Update income with growth factor
    return person.income



print("\nReady to continue.")

#%%
# SAMPLE CODES to try out the model
utopModel = myModel( { "gender": False, "ethnic": False } ) # no bias Utopia model
biasModel = myModel( { "gender": True, "ethnic": True } ) # bias, flawed, real world model

print("\nReady to continue.")

#%%
# Now try the two models on some versions of different people. 
# See what kind of range you can get. Plato is here for you as an example.
# industry: 0-leisure n hospitality, 1-retail , 2- Education 17024, 3-Health, 4-construction, 5-manufacturing, 6-professional n business, 7-finance
# gender: 0-female, 1-male
# marital: 0-never, 1-married, 2-divorced, 3-widowed
# ethnic: 0, 1, 2 
# age: 30-60, although there is no hard limit what you put in here.
# income: no real limit here.

months = 12 # Try months = 1, 12, 60, 120, 360
# In the ideal world model with no bias
plato = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'utop: {utopModel.predictGrowthFactor(plato)}') # This is the current growth factor for plato
print(f'utop: {utopModel.predictIncome(plato)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'utop: {utopModel.predictFinalIncome(months,plato)}')
#
# If plato ever gets a raise, or get older, you can update the info with a dictionary:
# plato.update( { "age": 59, "education": 21, "marital": 1, "income": 130000 } )

# In the flawed world model with biases on gender and ethnicity 
aristotle = Person( { "age": 58, "education": 20, "gender": 1, "marital": 0, "ethnic": 2, "industry": 7, "income": 100000 } )
print(f'bias: {biasModel.predictGrowthFactor(aristotle)}') # This is the current growth factor for aristotle
print(f'bias: {biasModel.predictIncome(aristotle)}') # This is the income after 1 month
# Do the following line when your new function predictFinalIncome is ready
print(f'bias: {biasModel.predictFinalIncome(months,aristotle)}')

# Example of a custom person
bewketu = Person({
    "age": 45,
    "education": 18,
    "gender": 0,       # Female
    "marital": 1,      # Married
    "ethnic": 1,
    "industry": 5,     # Manufacturing
    "income": 75000    # Starting income
})

print("\nFor a bewketu Person in the Utopia Model:")
print(f"Growth Factor: {utopModel.predictGrowthFactor(bewketu):.6f}")
print(f"Income After 1 Month: {utopModel.predictIncome(bewketu):.2f}")
print(f"Income After {months} Months: {utopModel.predictFinalIncome(months, bewketu):.2f}")

# Example of a custom person
beza = Person({
    "age": 30,
    "education": 21,
    "gender": 0,       # Female
    "marital": 1,      # Married
    "ethnic": 1,
    "industry": 5,     # Manufacturing
    "income": 75000    # Starting income
})

print("\nFor a beza Person in the Utopia Model:")
print(f"Growth Factor: {utopModel.predictGrowthFactor(beza):.6f}")
print(f"Income After 1 Month: {utopModel.predictIncome(beza):.2f}")
print(f"Income After {months} Months: {utopModel.predictFinalIncome(months, beza):.2f}")

print("\nReady to continue.")

# %%[markdown]

#%% [markdown]
# # Evolution (Part III - 25%)
# 
# We want to let the 24k people in WORLD#2 to evolve, for 360 months. You can either loop them through, and 
# create a new income or incomeFinal variable in the dataframe to store the new income level after 30 years. Or if you can figure out a way to do 
# broadcasting the predict function on the entire dataframem that can work too. If you loop through them, you can also consider 
# using Person class to instantiate the person and do the calcuations that way, then destroy it when done to save memory and resources. 
# If the person has life changes, it's much easier to handle it that way, then just tranforming the dataframe directly.
# 
# We have just this one goal, to see what the world look like after 30 years, according to the two models (utopModel and biasModel). 
# 
# Remember that in the midterm, world1 in terms of gender and ethnic groups, 
# there were not much bias. Now if we let the world to evolve under the 
# utopia model utopmodel, and the biased model biasmodel, what will the income distributions 
# look like after 30 years?
# 
# Answer this in terms of distribution of income only. I don't care about 
# other utopian measures in this question here. 
# 

#%% 
# # Reverse Action (Part IV - 25%)
# 
# Now let us turn our attension to World 1, which you should have found in the midterm that 
# it is far from being fair from income perspective among gender and ethnic considerations. 
# 
# Let us now put in place some policy action to reverse course, and create a revser bias model:
revbiasModel = myModel( { "gender": -1, "ethnic": -1 } ) # revsered bias, to right what is wronged gradually.

# If we start off with Word 1 on this revbiasModel, is there a chance for the world to eventual become fair like World #2? If so, how long does it take, to be fair for the different genders? How long for the different ethnic groups? 

# If the current model cannot get the job done, feel free to tweak the model with more aggressive intervention to change the growth rate percentages on gender and ethnicity to make it work. 

#%%

# newWorld2 = # world2.
def simulate(n, row):
  newPerson = Person(row)
  revbiasModel.predictFinalIncome(n, newPerson) # newPerson.income
  return newPerson.income
  

newlist = world2.apply( simulate() )

world2.income = newlist 