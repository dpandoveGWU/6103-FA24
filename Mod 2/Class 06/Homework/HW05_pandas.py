# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# # HW pandas 
# ## By: xxx
# ### Date: xxxxxxx
#

#%% [markdown]
#
#%%
# Step/Question 0, try reading the data file and make it a dataframe this time
import os
import numpy as np
import pandas as pd
import rfit

dats = rfit.dfapi('Dats_grades')
print("\nReady to continue.")

rfit.dfchk(dats)

# What are the variables in the df? 
# What are the data types for these variables?

#%% 
# Question 1
# The file has grades for a DATS class. Eight homeworks (out of 10 each), 2 quizzes (out of 100 each), and 2 projects (out of 100 each)
# Find out the class average for each item (HW, quiz, project)
# Hint, use .mean() function of pandas dataframe

#%%
# Question 2
# create a new column right after the last hw column, to obtain the average HW grade.
# use column name HWavg. Make the average out of the total of 100.
# Hint: use .iloc to select the HW columns, and then use .mean(axis=1) to find the row average

#%%
# Question 3
# The course total = 30% HW, 10% Q1, 15% Q2, 20% Proj1, 25% Proj2. 
# Calculate the total and add to the df as the last column, named 'total', out of 100 max.

#%%
# Question 4
# Now with the two new columns, calculate the class average for everything again. 

#%%
# Question 5
# Save out your dataframe as a csv file
# import os


#%%
# Question 6
# In one of the previous hw, we wrote a function to convert course total to letter grades. You can use your own code here.
def find_grade(total):
  # write an appropriate and helpful docstring
  """
  convert total score into grades
  :param total: 0-100 
  :return: str
  """
  # use conditional statement to set the correct grade
  pass   

#%%
# Question 7
# Let us create one more column for the letter grade, just call it grade.



#%%
# I would like to make the column headers/index multi-levels. 
# Let us achieve this in several steps. 
# First, let us create a student id column, so that each row is distinguishable 
# after we pull it apart. Use this 'stuId' as the new index
#
#%%
# Question 8a
# Now, filter out the the HW columns, and also keep the 'stuId' column
# Use whatever method you are comfortable with.
# Save this datafame as datshw.
# It should look like this:
# 	H1	H2	H3	H4	H5	H6	H7	H8	HWavg
# stuId									
# 1	10.0	10.0	10.0	10.0	9.5	9.5	10.0	10.0	98.75
# 2	10.0	10.0	10.0	9.9	9.5	9.0	10.0	10.0	98.00
# 3	10.0	10.0	9.5	10.0	9.5	10.0	10.0	10.0	98.75


#%%
# Question 8b
# Now, add one more level for the headers. So that the column headers 
# are ('HW', 'H1' ), ('HW', 'H2' ), etc 
# One method is add a column, simply call it 'type', with values = 'HW'. Remember that  pivot will 
# take the column values ('HW' here), and make it the new column header.
# The dataframe should look like this:
# H1	H2	H3	H4	H5	H6	H7	H8	HWavg	type
# stuId										
# 1	10.0	10.0	10.0	10.0	9.5	9.5	10.0	10.0	98.75	HW
# 2	10.0	10.0	10.0	9.9	9.5	9.0	10.0	10.0	98.00	HW
# 3	10.0	10.0	9.5	10.0	9.5	10.0	10.0	10.0	98.75	HW


#%%
# Question 8c
# Now, set composite index with 'stuId', 'type'. Use inplace = True
# The dataframe should look like this:
# H1	H2	H3	H4	H5	H6	H7	H8	HWavg
# stuId	type									
# 1	HW	10.0	10.0	10.0	10.0	9.5	9.5	10.0	10.0	98.75
# 2	HW	10.0	10.0	10.0	9.9	9.5	9.0	10.0	10.0	98.00
# 3	HW	10.0	10.0	9.5	10.0	9.5	10.0	10.0	10.0	98.75

# 
#%%
# Question 8d
# Now, stack all the columns into a long dataframe. What is the index now? It 
# should be a three-level index, with names names=['stuId', 'type', None], length=837) 
# The new column has no name, where it records whether it is H1, or H2, etc. 

# Use the stack() method here.

# You should rename the missing index column so that we can call it later. 
# Say call it 'colName', with values 'H1', 'H2', etc.
# Also give a name for the data column, call it 'vals'
# The resulting stack object is a Pandas Series, and should look like this:
# stuId  type  colName
# 1      HW    H1         10.0
#              H2         10.0
#              H3         10.0
# The name 'vals' for this Pandas series does not show here, but you can 
# check that  datshwstack.name should equal to 'vals'






print(datshwstack.index)

#%%
# Question 8e
# Now we can pivot this table back into multi-index columns with ('HW', 'H1'), etc 
# Will need to convert the pandas series into pandas dataframe first. 
# The df should look like this:
# type       HW                                                 
# colName    H1    H2    H3    H4    H5    H6    H7    H8  HWavg
# stuId                                                         
# 1        10.0  10.0  10.0  10.0   9.5   9.5  10.0  10.0  98.75
# 2        10.0  10.0  10.0   9.9   9.5   9.0  10.0  10.0  98.00
# 3        10.0  10.0   9.5  10.0   9.5  10.0  10.0  10.0  98.75



print(datshwfinal.index)

#%%
# If we continue to do this for Quiz and Project columns, we can merge 
# these back together to obatin a dataframe with multi-index columns, which 
# is easy to filter out hw grades from quiz grades, etc.


#%%
