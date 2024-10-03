# In labor economics, many curious questions stem from the relationship
# between wage and education. 
# We explore the famous `wage1` dataset from the `wooldridge` package.

# %%
# Import the data importer function from `wooldridge`
#pip install wooldridge
from wooldridge import dataWoo

# %%
# Import the other relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as py
#import statsmodels.api as sm
from scipy import stats

# %%
#Q1
# Read the description 
# Explain what you gather from description about the variables `wage`, `educ`, 
# and `female`
dataWoo('wage1', description = True)

# Answer:
# wage: 
# educ: 
# female: 
# %%
#Q2
# Now, load the dataset into a dataframe wage 1



# %%
#Q3
# Print the first and last few rows


# %%
# Basic sanity check for values that are not usable
# Standard quick checks
wage1.describe()
wage1.info()

#Q4
#Write Observations here:




# %%
# `id` looks like the unique identifier. 
# Let's check to see if it is - if so we'll set it as the row index.
print(len(pd.unique(wage1['id'])) == len(wage1))
#Q5
# Write conclusion of this analysis:



# %%
#Q6
# Set `id` field as index and then print the first few rows



# %%
# Now check0 the variables of variables of interest for null values
wage1['wage'].isnull().values.sum()
wage1['educ'].isnull().values.sum()
wage1['female'].isnull().values.sum()
#Q7
# Write your observations here:



# %%
#Q8
# If NA values are detected, drop them.
# Don't drop if the NA values are not affecting the columns of interest


