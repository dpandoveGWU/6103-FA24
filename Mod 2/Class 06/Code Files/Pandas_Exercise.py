# In labor economics, many curious questions stem from the relationship
# between wage and education. 
# We explore the famous `wage1` dataset from the `wooldridge` package.

# Group Members:
# 1) Wali
# 2) Naiska
# 3) Sharon
# 4) Suraj 

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

# Answer: Gives us a description of each column and what it holds
# wage: average hourly earnings
# educ: years of education
# female: has binary value. If 1 then female else not
# %%
#Q2
# Now, load the dataset into a dataframe wage 1
wage1 = dataWoo('wage1')


# %%
#Q3
# Print the first and last few rows
wage1.head()

# %%
# Basic sanity check for values that are not usable
# Standard quick checks
wage1.describe()
#wage1.info()

#Q4
#Write Observations here:
# All columns in the dataset seem to have no null values
# wage is of type float rest are int
# data has 24 columns
# data has 526 rows
# mean experience is quite high. Maybe a lot of outliers but need to check.

# %%
# (Question 4 b) Create a column named  id for unique index
wage1.reset_index(inplace=True)
wage1.rename(columns={'index': 'id'}, inplace=True)


# %%
wage1.head()
# %%
# `id` looks like the unique identifier. 
# Let's check to see if it is - if so we'll set it as the row index.
print(len(pd.unique(wage1['id'])) == len(wage1))

#Q5
# Write conclusion of this analysis:

# (Answer): id is indeed a unique column as the print statement returns true

# %%
#Q6
# Set `id` field as index and then print the first few rows

wage1.set_index('id', inplace=True)
wage1.head()

# %%
# Now check0 the variables of variables of interest for null values
wage1['wage'].isnull().values.sum()
wage1['educ'].isnull().values.sum()
wage1['female'].isnull().values.sum()
#Q7
# Write your observations here:
# (Answer): There are no nans in these columns


# %%
#Q8
# If NA values are detected, drop them.
# Don't drop if the NA values are not affecting the columns of interest
# (Answer): there are no null values so nothing will be dropped
