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

# Set plot style for consistency
sns.set(style="whitegrid")

# 1a. Histogram on age, with a stacked male-female series
plt.figure(figsize=(10, 6))
# Plot age histogram for males
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

gender_proportion, survival_proportion


# ### c. pie chart for “Ticketclass”  
# %%

# Calculate the proportion for each ticket class
ticket_class_proportion = titanic['pclass'].value_counts(normalize=True)

# Plot the pie chart for ticket class
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

# # Step 3: Encode categorical variables ('sex' and 'embarked')
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
result.summary()


# parch and fare have higher p values and they are insignificant.let's drop them
X_refined = X.drop(columns=['parch', 'fare'])
import statsmodels.api as sm
logit_model_refined = sm.Logit(y, X_refined)
result_refined = logit_model_refined.fit()
print(result_refined.summary())

# %%
#  Compare Log-Likelihood and AIC
print("\nComparison of Metrics:")
print(f"Original Model - Log-Likelihood: {result.llf}, AIC: {result.aic}")
print(f"Refined Model - Log-Likelihood: {result_refined.llf}, AIC: {result_refined.aic}")

# %%
""""""
# %%
# ## Question 3  
# Interpret your result. What are the factors and how do they affect the chance of survival (or the survival odds ratio)? What is the predicted probability of survival for a 30-year-old female with a second class ticket, no siblings, 3 parents/children on the trip? Use whatever variables that are relevant in your model.  

# ## Question 4  
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)


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
# ## Question 6  
# Using the SciKitLearn library, build a logistic regression model overall (not individual team or kicker) to predict the chances of a successful field goal. What variables do you have in your model? 
# 
# ## Question 7  
# Someone has a feeling that home teams are more relaxed and have a friendly crowd, they should kick better field goals. Use your model to find out if that is subtantiated or not. 
# 
#  
# ## Question 8    
# From what you found, do home teams and road teams have different chances of making a successful field goal? If one does, is that true for all distances, or only with a certain range?
# 


# %%
# titanic.dropna()


# %%