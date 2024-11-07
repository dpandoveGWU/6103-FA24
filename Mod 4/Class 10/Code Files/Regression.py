#%%
# First read in the datasets. One Graduate school admission dataset, one Titanic survival dataset
import pandas as pd
dfadmit = pd.read_csv('gradAdmit.csv')



#%%
# quick plots
import matplotlib.pyplot as plt
# add color
import numpy as np
admitcolors = np.where(dfadmit['admit']==1,'g','r')
# admitcolors[dfadmit['admit']==0] = 'r'
# admitcolors[dfadmit['admit']==1] = 'g'

print("\nReady to continue.")

#%% 
# Plot, Pandas style (or seaborn sns)
dfadmit.plot(x="gre", y="gpa", kind="scatter", color=admitcolors)
plt.xlabel("GRE score")
plt.ylabel("GPA")
plt.title("GPA vs GRE")
# plt.savefig(filepath, dpi=96)
plt.show()

#%%
# OR Matplotlib focused
plt.scatter(dfadmit.gre,dfadmit.gpa, color=admitcolors)  
plt.xlabel("GRE score")
plt.ylabel("GPA")
plt.title("GPA vs GRE")
plt.show()


#%%
# Note that the plot function here by default is a line plot 
plt.plot(dfadmit.gre,dfadmit.gpa, 'r-o')  # red, solid line, circle dots
plt.show()
#The plot will show GRE scores on the x-axis and GPA scores on the y-axis, with a red line connecting the points and red circles at each data point. It gives a quick visual sense of how GRE scores vary with GPA values in this dataset.

#%%
# OR
# more object-oriented style
fig, axis = plt.subplots()
axis.plot(dfadmit.gre, dfadmit.gpa, color='g', linestyle="", marker="o", markersize=3)
plt.xlabel("GRE score")
plt.ylabel("GPA")
plt.title("GPA vs GRE")
# plt.savefig(filepath, dpi=96)
plt.show()

#%%
# easier to add jittering
fig, axis = plt.subplots(2,2)
axis[0,0].plot(dfadmit.gre, dfadmit.gpa, color='r', linestyle="", marker="^", markersize=3, alpha=0.3)
# axis[0,0].xlabel("GRE score")
# axis[0,0].ylabel("GPA")
axis[0,0].set_title("plain")
axis[0,0].xaxis.set_ticklabels([]) # get rid of x tick marks for clarity here

axis[0,1].plot(dfadmit.gre + np.random.uniform(0,10, size=dfadmit.shape[0] ), dfadmit.gpa, color='g', linestyle="", marker="o", markersize=3, alpha=0.3)
axis[0,1].set_title("jitter gre")
axis[0,1].xaxis.set_ticklabels([]) # get rid of x tick marks for clarity here

axis[1,0].plot(dfadmit.gre, dfadmit.gpa + np.random.uniform(0,.1, size=dfadmit.shape[0]), color='b', linestyle="", marker="+", markersize=3, alpha=0.4)
axis[1,0].set_title("jitter gpa")

axis[1,1].plot(dfadmit.gre + np.random.uniform(0,10, size=dfadmit.shape[0] ), dfadmit.gpa + np.random.uniform(0,.1, size=dfadmit.shape[0]), color='#555555', linestyle="", marker="x", markersize=3, alpha=0.5)
axis[1,1].set_title("jitter both")

# plt.xlabel("GRE score")
# plt.ylabel("GPA")
# plt.title("GPA vs GRE")
plt.savefig("quad_figs.png", dpi=96)
plt.show()

#%% 
# seaborn sns
import seaborn as sns
sns.scatterplot(data=dfadmit, x='gre', y='gpa')
sns.despine()


#%% 
# seaborn sns
# import seaborn as sns
sns.regplot(x='gre', y ='gpa', data=dfadmit, fit_reg = False, x_jitter=10, scatter_kws={'alpha': 0.3, 's': 3 } )
sns.despine()
# easy
# lack some minor control such as what distribution to use
# can also use subplots, with different set of syntax

#%% 
# seaborn sns
# import seaborn as sns
sns.regplot(x='gre', y='gpa', data=dfadmit, fit_reg = True, x_jitter=10, scatter_kws={'alpha': 0.3, 's': 3 }, line_kws = {'color':'red', 'label':'LM fit'})
sns.despine()

#%% 
# seaborn sns
# import seaborn as sns
#
# https://datascience.stackexchange.com/questions/44192/what-is-the-difference-between-regplot-and-lmplot-in-seaborn
# regplot() performs a simple linear regression model fit and plot. lmplot() combines regplot() and FacetGrid.
# The FacetGrid class helps in visualizing the distribution of one variable as well as the relationship between multiple variables separately within subsets of your dataset using multiple panels.
# lmplot in particular has the hue option
sns.lmplot(x='gre',y= 'gpa', data=dfadmit, hue='admit', fit_reg = False, x_jitter=10, scatter_kws={'alpha': 0.3, 's': 3 })
sns.despine()


#%% 
# seaborn sns
# import seaborn as sns
# lmplot also allows multiple series with different markers.
sns.lmplot(x='gre', y='gpa', data=dfadmit, hue='admit', markers=["o", "x"], fit_reg = True, x_jitter=10, scatter_kws={'alpha': 0.4, 's': 8 })
sns.despine()


#%% 
# seaborn sns
# color by rank
# import seaborn as sns
sns.lmplot(x='gre',y= 'gpa', data=dfadmit, hue='rank', markers=['o', 'x', '^', 's'], fit_reg = False, x_jitter=10, scatter_kws={'alpha': 0.4, 's': 8 })
# markers=['o', 'x', 's','^','p','+','d']
sns.despine()

#%%
# Question, 
# How many dimensions we can visualize?



#%% 
# color by rank
# Let also try pandas, 
# need to create the color label for each data point ourselves, 
# but we can have color and shape separate
rankcolors = np.where(dfadmit['rank']==1,'r','-') # initialize the vector as well
# rankcolors[dfadmit['rank']==1] = 'r'
rankcolors[dfadmit['rank']==2] = 'g'
rankcolors[dfadmit['rank']==3] = 'b'
rankcolors[dfadmit['rank']==4] = 'yellow'

# and use different shape for admit 0 and 1
ax1 = dfadmit[dfadmit.admit==0].plot(x="gre", y="gpa", kind="scatter", color=rankcolors[dfadmit.admit==0], marker='o', label='rejected')
dfadmit[dfadmit.admit==1].plot(x="gre", y="gpa", kind="scatter", color=rankcolors[dfadmit.admit==1], marker='+', label='admitted', ax = ax1)
# dfadmit.plot(x="gre", y="gpa", kind="scatter", color=rankcolors, marker='+')
plt.legend(loc='upper left')
plt.xlabel("GRE score")
plt.ylabel("GPA")
plt.title("GPA vs GRE")
# plt.savefig(filepath, dpi=96)
plt.show()


#%% 
# color by rank
# Try Matplotlib, so we can add jittering
fig, axis = plt.subplots()
for admitval, markerval in { 0: "o" , 1: "+" }.items() : # rejected (admit==0), use 'o', admitted (admit==1), use '+'
  for cindex, cvalue in {1: 'r', 2: 'g', 3: 'b', 4: 'yellow' }.items() : # the ranks and colors
    thisdf = dfadmit[dfadmit.admit==admitval] # first filter out admitted or rejected
    thisdf = thisdf[thisdf['rank'] == cindex] # then filter out one rank at a time.
    print(thisdf.shape)
    axis.plot(thisdf.gre + np.random.uniform(0,10, size=thisdf.shape[0] ), 
              thisdf.gpa, 
              color=cvalue, 
              linestyle="", 
              marker=markerval, 
              markersize=3, 
              alpha=0.3
    )

plt.xlabel("GRE score")
plt.ylabel("GPA")
plt.title("GPA vs GRE")
# plt.savefig(filepath, dpi=96)
plt.show()

#%%
# Now, your turn. Try some sensible plots with the Titanic dataset. 
# How would you visualize the relations between survived, age, sex, fare, embarked? 
# You do not need to use all of them in a single plot. What variables make the most sense to you, 
# in terms of finding out who survived, and who didn't.
#
dftitan = pd.read_csv('Titanic.csv')
# perform a quick clean up on age NAs


#%% 
# Now LINEAR REGRESSION
# 1. Describe the model → ols()
# 2. Fit the model → .fit()
# 3. Summarize the model → .summary()
# 4. Make model predictions → .predict()


#%%
# FORMULA based
from statsmodels.formula.api import ols
modelGreGpa = ols(formula='gre ~ gpa', data=dfadmit)
print( type(modelGreGpa) )

#%%
modelGreGpaFit = modelGreGpa.fit()
print( type(modelGreGpaFit) )
print( modelGreGpaFit.summary() )

# From the summary, try to get as much info as we can
# Df Residuals (# total observations minus Df Model minus 1)
# Df Model (# of x variables)
# R-squared, what does that mean?
# Adj R-squared
# F-statistics
# Prob (F-statistics), ie. p-value for F-statistics
# Log-Likelihood
# AIC (model eval)
# BIC (model eval)

# coef
# std err
# t
# P>|t|, aka p-value for the coefficient significance
# 95% confidence intervals

# Omnibus - close to zero means residuals are normally distributed
# Prob(Omnibus) - close to 1 means residuals are normally distributed
# skew (positive is right tailed, negative is left)
# Kurtosis (tailedness, normal dist = 3, less than 3 is fatter tail, and flat top.)

print("\nReady to continue.")

#%% 
import pandas as pd
modelpredicitons = pd.DataFrame( columns=['gre_GpaLM'], data= modelGreGpaFit.predict(dfadmit.gpa)) 
# use the original dataset gpa data to find the expected model values
print(modelpredicitons.shape)
print( modelpredicitons.head() )

print("\nReady to continue.")

#%%
# Next let us try more variables, and do it in a combined step
modelGreGpaRankFit = ols(formula='gre ~ gpa + rank', data=dfadmit).fit()

print( type(modelGreGpaRankFit) )
print( modelGreGpaRankFit.summary() )

modelpredicitons['gre_GpaRankLM'] = modelGreGpaRankFit.predict(dfadmit)
print(modelpredicitons.head())

print("\nReady to continue.")

#%%
# And let us check the VIF value (watch out for multicollinearity issues)
# Import functions
# from statsmodels.stats.outliers_influence import variance_inflation_factor

# # Get variables for which to compute VIF and add intercept term
# X = dfadmit[['gpa', 'rank']]
# X['Intercept'] = 1

# # Compute and view VIF
# vif = pd.DataFrame()
# vif["variables"] = X.columns
# vif["VIF"] = [ variance_inflation_factor(X.values, i) for i in range(X.shape[1]) ] # list comprehension

# # View results using print
# print(vif)

# print("\nReady to continue.")

#%% [markdown]
# But rank really should be categorical. 
# 
# # Patsy coding
# 
# * Strings and booleans are automatically coded
# * Numerical → categorical
#   * C() function
#   * level 0 → (0,0,0,...)
#   * level 1 → (1,0,0,...)
#   * level 2 → (0,1,0,...)
# * Reference group
#   * Default: first group
#   * Treatment
#   * levels

#%%
modelGreGpaCRankFit = ols(formula='gre ~ gpa + C(rank)', data=dfadmit).fit()
print( modelGreGpaCRankFit.summary() )

modelpredicitons['gre_GpaCRankLM'] = modelGreGpaCRankFit.predict(dfadmit)
#modelGreGpaCRankFit.predict(dfadmit): Uses the fitted model to predict GRE scores for each sample in dfadmit.
#modelpredicitons['gre_GpaCRankLM']: Stores these predictions in a new column called gre_GpaCRankLM within a DataFrame named modelpredictions.

print(modelpredicitons.head())
#modelpredicitons.head(): Displays the first five rows of the modelpredictions DataFrame, allowing you to inspect the predicted values for GRE (stored in gre_GpaCRankLM).

print("\nReady to continue.")

#%%
# Next try some interaction terms

# 
# formula = 'y ~ x1 + x2'
# C(x1) : treat x1 as categorical variable
# -1 : remove intercept
# x1:x2 : an interaction term between x1 and x2
# x1*x2 : an interaction term between x1 and x2 and the individual variables
# np.log(x1) : apply vectorized functions to model variables

modelGreGpaXCRankFit = ols(formula='gre ~ gpa * C(rank)', data=dfadmit).fit()
print( modelGreGpaXCRankFit.summary() )
modelpredicitons['gre_GpaXCRankLM'] = modelGreGpaXCRankFit.predict(dfadmit)
print(modelpredicitons.head())

# This is essentially four different models for the four ranks of schools.

# QUESTION: Can you build a model which encompass four models for the four different schools 
# with the same slope (for gpa) but allow for different intercepts?

print("\nReady to continue.")

#%% [markdown]
# # Logistic Regressions
#
# link function in glm
# https://www.statsmodels.org/stable/glm.html#families
# Gaussian(link = sm.families.links.identity) → the default family
# Binomial(link = sm.families.links.logit)
# probit, cauchy, log, and cloglog
# Poisson(link = sm.families.links.log)
# identity and sqrt


#%% [markdown]
#
# # Maximum Likelihood Estimation
#
# Likelihood vs Probability
# Conditional Probability: P (outcome A∣given B)
# Probability: P (data∣model)
# Likelihood: L(model∣data)
#
# If the error distribution is normal, and we chose to use a square (Euclidean) 
# distance metric, then OLS and MLE produces the same result.


#%%
import statsmodels.api as sm  # Importing statsmodels
# import statsmodels.formula.api as smf  # Support for formulas
# from statsmodels.formula.api import glm   # Use glm() directly

# 1. Describe the model → glm()
# 2. Fit the model → .fit()
# 3. Summarize the model → .summary()
# 4. Make model predictions → .predict()

# 1. Describe the model → glm()

# Two of the available styles:
# ARRAY based
# import statsmodels.api as sm
# X = sm.add_constant(X)
# model = sm.glm(y, X, family)

# FORMULA based (we had been using this for ols)
from statsmodels.formula.api import glm
# model = glm(formula, data, family)

modelAdmitGreLogit = glm(formula='admit ~ gre', data=dfadmit, family=sm.families.Binomial())

#%%
modelAdmitGreLogitFit = modelAdmitGreLogit.fit()
print( modelAdmitGreLogitFit.summary() )
modelpredicitons['admit_GreGpaLogit'] = modelAdmitGreLogitFit.predict(dfadmit)
# print(modelpredicitons.head())
# dm.dfChk(modelpredicitons)

print("\nReady to continue.")

#%% [markdown]
# # Deviance
# Formula
# D = −2LL(β)
# * Measure of error
# * Lower deviance → better model fit
# * Benchmark for comparison is the null deviance → intercept-only model / constant model
# * Evaluate
#   * Adding a random noise variable would, on average, decrease deviance by 1
#   * Adding k predictors to the model deviance should decrease by more than k

#%%
# The deviance of the model was 486.06 (or negative two times Log-Likelihood-function)
# df = 398 
print(-2*modelAdmitGreLogitFit.llf)
# Compare to the null deviance
print(modelAdmitGreLogitFit.null_deviance)
# 499.98  # df = 399 
# A decrease of 14 with just one variable. That's not bad. 
# 
# Another way to use the deviance value is to check the chi-sq p-value like this:
# Null model: chi-sq of 399.98, df = 399, the p-value is 0.000428 (can use scipy.stats.chisquare function) 
# Our model: chi-sq of 486.06, df = 398, the p-value is 0.001641
# These small p-values (less than 0.05, or 5%) means reject the null hypothesis, which means the model is not a good fit with data. We want higher p-value here. Nonetheless, the one-variable model is a lot better than the null model.



print("\nReady to continue.")

#%%
# Now with more predictors
modelAdmitAllLogit = glm(formula='admit ~ gre+gpa+C(rank)', data=dfadmit, family=sm.families.Binomial())
modelAdmitAllLogitFit = modelAdmitAllLogit.fit()
print( modelAdmitAllLogitFit.summary() )
modelpredicitons['admit_GreAllLogit'] = modelAdmitAllLogitFit.predict(dfadmit)
# print(modelpredicitons.head())
# dm.dfChk(modelpredicitons)

# QUESTION: Is this model separable into four models for each rank with the 
# same "intercept" or "slopes"? 
# How can you generalize it to a more general case?

print("\nReady to continue.")

#%%
# Testing
modelAdmitTestLogit = glm(formula='admit ~ gre+gpa+C(rank)+gre*C(rank)', data=dfadmit, family=sm.families.Binomial())
modelAdmitTestLogitFit = modelAdmitTestLogit.fit()
print( modelAdmitTestLogitFit.summary() )

#%%
# To interpret the model properly, it is handy to have the exponentials of the coefficients
np.exp(modelAdmitAllLogitFit.params)
np.exp(modelAdmitAllLogitFit.conf_int())

print("\nReady to continue.")

#%%
# Confusion matrix
# Define cut-off value
cut_off = 0.3
# Compute class predictions
modelpredicitons['classLogitAll'] = np.where(modelpredicitons['admit_GreAllLogit'] > cut_off, 1, 0)
print(modelpredicitons.classLogitAll.head())
#Setting a Custom Threshold: By using a lower threshold (0.3), the model classifies a sample as admitted with a relatively low probability of admission, which may be desirable in certain applications where missing an admission prediction is costlier than a false positive.
#Resulting Class Predictions: The resulting classLogitAll column contains binary classifications based on the threshold, showing whether each sample is classified as admitted (1) or not (0). This can be used to evaluate the impact of the custom threshold on model performance, sensitivity, and specificity.
#
# Make a cross table
print(pd.crosstab(dfadmit.admit, modelpredicitons.classLogitAll,
rownames=['Actual'], colnames=['Predicted'],
margins = True))
#
#
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

print("\nReady to continue.")

#%%
# Now try the Titanic Dataset, and find out the survival chances from different predictors.

