#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rfit

# Part I
titanic = pd.read_csv('Titanic.csv', index_col='id')

# Part II
nfl = pd.read_csv('nfl2008_fga.csv')
nfl.dropna(inplace=True)

#%% [markdown]

# # Part I  
# Titanic dataset - statsmodels
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | survival | Survived or not | 0 = No, 1 = Yes |  
# | pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |  
# | sex | Gender / Sex |  |  
# | age | Age in years |  |  
# | sibsp | # of siblings / spouses on the Titanic |  |  
# | parch | # of parents / children on the Titanic |  |  
# | ticket | Ticket number (for superstitious ones) |  |  
# | fare | Passenger fare |  |  
# | embarked | Port of Embarkation | C: Cherbourg, Q: Queenstown, S: Southampton  |  
# 
#%%
# ## Question 1  
# With the Titanic dataset, perform some summary visualizations:  
# 
# %%
# ### a. Histogram on age. Maybe a stacked histogram on age with male-female as two series if possible
# plot style for consistency
sns.set(style="whitegrid")
# 1a. Histogram on age, with a stacked male-female series
plt.figure(figsize=(10, 6))
# age histogram for males
sns.histplot(data=titanic[titanic['sex'] == 'male'], x='age', bins=30, color='blue', label='Male', kde=False)
# Overlay age histogram for females
sns.histplot(data=titanic[titanic['sex'] == 'female'], x='age', bins=30, color='pink', label='Female', kde=False)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Histogram of Age with Male-Female Stacked")
plt.legend()
plt.show()

#%%
# ### b. proportion summary of male-female, survived-dead  
# Proportion of male and female passengers
gender_proportion = titanic['sex'].value_counts(normalize=True)
# Proportion of survived and dead passengers
survival_proportion = titanic['survived'].value_counts(normalize=True)
print("proportion of male to female is:",gender_proportion,"\n proportion of surviva-dead is:" ,survival_proportion)


 
# %%
# ### c. pie chart for “Ticketclass” 
# proportion for each ticket class
ticket_class_proportion = titanic['pclass'].value_counts(normalize=True)
# pie chart for ticket class
plt.figure(figsize=(8, 6))
plt.pie(ticket_class_proportion, labels=ticket_class_proportion.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title("Proportion of Ticket Class (1st, 2nd, 3rd)")
plt.show()


  
# %%
# ### d. A single visualization chart that shows info of survival, age, pclass, and sex.
g = sns.FacetGrid(titanic, col="pclass", row="sex", hue="survived", height=4, aspect=1.2)
g.map(sns.histplot, "age", kde=False, bins=20)
g.add_legend(title="Survived")
g.set_axis_labels("Age", "Count")
g.set_titles("{row_name} - Class {col_name}")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Survival Distribution by Age, Pclass, and Sex")
plt.show()
# %%
# ## Question 2  
# Build a logistic regression model for survival using the statsmodels library. As we did before, include the features that you find plausible. Make sure categorical variables are use properly. If the coefficient(s) turns out insignificant, drop it and re-build.  
titanic_data = pd.read_csv('Titanic.csv')
#Drop unnecessary columns ('id', 'ticket')
titanic_data.drop(columns=['id', 'ticket'], inplace=True)
# Display the number of missing values for each column
missing_values = titanic_data.isnull().sum()
print(missing_values)
# Handle missing values for age and embarked
# Fill missing 'age' with the median value
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)
# Fill missing 'embarked' with the most common value ('S')
titanic_data['embarked'].fillna(titanic_data['embarked'].mode()[0], inplace=True)
#Encode categorical variables ('sex' and 'embarked')
# Encode 'sex': male = 1, female = 0
titanic_data['sex'] = titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
# Encode 'embarked': C = 0, Q = 1, S = 2
titanic_data['embarked'] = titanic_data['embarked'].apply(lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))
# Display the first few rows to verify
titanic_data.head()


# %%
import statsmodels.api as sm

# Define the target variable 'y' and feature set 'X'
X = titanic_data.drop(columns=['survived'])
y = titanic_data['survived']
# Add a constant to the feature set (intercept) and build the model
X = sm.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()
# summary of the model
print("model result before refined",result.summary())

# parch and fare have higher p values and they are insignificant.let's drop them
X_refined = X.drop(columns=['parch', 'fare'])
import statsmodels.api as sm
logit_model_refined = sm.Logit(y, X_refined)
result_refined = logit_model_refined.fit()
print("\n model result after refined\n",result_refined.summary())
#  Compare Log-Likelihood and AIC
print("\nComparison of Metrics:")
print(f"Original Model - Log-Likelihood: {result.llf}, AIC: {result.aic}")
print(f"Refined Model - Log-Likelihood: {result_refined.llf}, AIC: {result_refined.aic}")

# %%
"""The refined model, even with a slightly worse Log-Likelihood, is considered better overall because it achieves a lower AIC.  
This suggests that dropping the insignificant variables (`parch` and `fare`) made the model simpler without sacrificing much predictive power.  
However, the change in AIC is very small, so the improvement is marginal. Both models perform quite similarly.
 is marginal. Both models perform quite similarly."""

# %%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  
"""Passenger Class (pclass=-1.1455): Passengers in higher classes (1st class) had a significantly better chance of survival. The odds of survival decrease as the class number increases (i.e., moving from 1st to 3rd class).
Sex (sex=-2.7158): Being female greatly increased the likelihood of survival compared to males. The model shows that males had much lower odds of survival.
Age (age=-0.0389): Older passengers were less likely to survive. For each additional year of age, the odds of survival decreased slightly.
Number of Siblings/Spouses (sibsp=-0.3356): Having more siblings or spouses onboard slightly reduced the chances of survival.
Embarkation Point (embarked=-0.2352): Passengers' survival odds varied slightly depending on their port of embarkation, with certain embarkation points being associated with lower survival odds."""

# Coefficients from the logistic regression model
intercept = 5.4256
coef_pclass = -1.1455
coef_sex = -2.7158
coef_age = -0.0389
coef_sibsp = -0.3356
coef_embarked = -0.2352

#features for the given individual
pclass = 2
sex = 0  # Female
age = 30
sibsp = 0
parch = 3
embarked = 0  

# Calculate the log-odds
log_odds = (intercept + 
            coef_pclass * pclass + 
            coef_sex * sex + 
            coef_age * age + 
            coef_sibsp * sibsp + 
            coef_embarked * embarked)

# Convert log-odds to probability
probability = np.exp(log_odds) / (1 + np.exp(log_odds))
print(f"Predicted probability of survival: {probability:.2f}")

# This indicates that a 30-year-old female in 2nd class with no siblings and 3 parents/children has an approximately 88% chance of survival according to your refined logistic regression model.

# %%
# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

# cut-off values
cut_off_values = [0.3, 0.5, 0.7]

# Function to evaluate the model
def evaluate_model(model, X, y, cut_off):
    y_prob = model.predict(X)
    y_pred = (y_prob >= cut_off).astype(int)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    
    return accuracy, precision, recall

# Train the logistic regression model on the refined features
logit_model_refined = sm.Logit(y, X_refined)
result_refined = logit_model_refined.fit()

# Evaluate the model for different cut-off values
for cut_off in cut_off_values:
    accuracy, precision, recall = evaluate_model(result_refined, X_refined, y, cut_off)
    print(f"Cut-off: {cut_off}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("-" * 40)



#%%[markdown]
# # Part II  
# NFL field goal dataset - SciKitLearn
# 
# | Variable | Definition | Key/Notes  |  
# | ---- | ---- | ---- |   
# | AwayTeam | Name of visiting team | |  
# | HomeTeam | Name of home team | |  
# | qtr | quarter | 1, 2, 3, 4 |  
# | min | Time: minutes in the game |  |  
# | sec | Time: seconds in the game |  |  
# | kickteam | Name of kicking team |  |  
# | distance | Distance of the kick, from goal post (yards) |  |  
# | timerem | Time remaining in game (seconds) |  |  
# | GOOD | Whether the kick is good or no good | If not GOOD: |  
# | Missed | If the kick misses the mark | either Missed |  
# | Blocked | If the kick is blocked by the defense | or blocked |  
# 
#%% 
# ## Question 5  
# With the nfl dataset, perform some summary visualizations.  
# 

nfl_data=pd.read_csv('nfl2008_fga.csv')
# Select relevant columns for analysis
columns_of_interest = ['distance', 'homekick', 'GOOD', 'Blocked', 'Missed']
nfl_cleaned = nfl_data[columns_of_interest].copy()

# Check for missing values
missing_values = nfl_cleaned.isnull().sum()
# Ensure only one outcome column ('GOOD') is present for simplicity in modeling
nfl_cleaned['unsuccessful'] = nfl_cleaned['Blocked'] | nfl_cleaned['Missed']
nfl_cleaned.drop(columns=['Blocked', 'Missed'], inplace=True)
# Verify the cleaned dataset
nfl_cleaned.head(), missing_values
# Set visualization style
sns.set(style="whitegrid")

#Distribution of field goal distances
plt.figure(figsize=(10, 6))
sns.histplot(nfl_cleaned['distance'], bins=30, kde=True, color="blue")
plt.title("Distribution of Field Goal Distances", fontsize=16)
plt.xlabel("Distance (yards)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

#Success rate by distance
plt.figure(figsize=(10, 6))
sns.lineplot(data=nfl_cleaned, x="distance", y="GOOD", estimator="mean", ci=None, color="green")
plt.title("Field Goal Success Rate by Distance", fontsize=16)
plt.xlabel("Distance (yards)", fontsize=12)
plt.ylabel("Success Rate", fontsize=12)
plt.ylim(0, 1)
plt.show()

#Success rate comparison - home vs away
plt.figure(figsize=(10, 6))
sns.barplot(data=nfl_cleaned, x="homekick", y="GOOD", ci=None, palette="viridis")
plt.title("Field Goal Success Rate: Home vs Away", fontsize=16)
plt.xlabel("Kicking Team (0 = Away, 1 = Home)", fontsize=12)
plt.ylabel("Success Rate", fontsize=12)
plt.ylim(0, 1)
plt.show()

# %%
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model?
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Features: 'distance', 'homekick'
# Target: 'GOOD' (1 = successful, 0 = unsuccessful)
X = nfl_cleaned[['distance', 'homekick']]
y = nfl_cleaned['GOOD']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Build and train the logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# model evaluation
y_pred = log_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
coefficients = log_model.coef_[0]
intercept = log_model.intercept_[0]

# Formatting the results
result = {
    "Model Coefficients": {
        "Distance Coefficient": coefficients[0],
        "Homekick Coefficient": coefficients[1],
    },
    "Intercept": intercept,
    "Model Accuracy": f"{accuracy * 100:.2f}%",
    "Classification Report": report
}

result

# %%
 ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Use your model to find out if that is subtantiated or not. 
from scipy.stats import ttest_ind

#logistic regression model
nfl_cleaned['predicted_prob'] = log_model.predict_proba(X)[:, 1]
#home and away predictions
home_probs = nfl_cleaned[nfl_cleaned['homekick'] == 1]['predicted_prob']
away_probs = nfl_cleaned[nfl_cleaned['homekick'] == 0]['predicted_prob']
#mean success probabilities for home and away
home_mean_prob = np.mean(home_probs)
away_mean_prob = np.mean(away_probs)
t_stat, p_value = ttest_ind(home_probs, away_probs, equal_var=False)
output = {
    "Home Mean Probability": home_mean_prob,
    "Away Mean Probability": away_mean_prob,
    "T-Test Statistic": t_stat,
    "P-Value": p_value
}
print(output)

"""Away teams have a significantly higher predicted probability of successfully kicking field goals compared to home teams. 
The small p-value suggests this difference is highly statistically significant, disproving the hypothesis that home teams perform better due to a friendly crowd."""

# %%
# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?
# 
distance_bins = [0, 30, 50, nfl_cleaned['distance'].max()]
distance_labels = ['0-30 yards', '31-50 yards', '>50 yards']
nfl_cleaned['distance_range'] = pd.cut(nfl_cleaned['distance'], bins=distance_bins, labels=distance_labels)

# Calculate success rates for home and away teams in each distance range
success_rates_by_range = nfl_cleaned.groupby(['distance_range', 'homekick'])['GOOD'].mean().unstack()
success_rates_by_range




# %%
"""Away teams have consistently higher success rates than home teams across all distance ranges. The disparity is particularly notable for field goals over 50 yards, where away teams outperform home teams by a wide margin. 
  This suggests that the difference in success is not limited to specific distances but is more pronounced for longer attempts.
  from all the above analysis that home teams are not better at kicking field goals, and in fact, away teams demonstrate higher success rates overall, particularly for long-distance kicks. 
  The idea of home advantage in this context is not supported."""


# %%