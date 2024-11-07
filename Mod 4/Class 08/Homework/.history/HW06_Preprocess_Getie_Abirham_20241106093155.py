# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

####### This part is optional and not a part of the assignment#######################
# Use the dataset "Happy.csv" is obtained from 
# https://gssdataexplorer.norc.org 
# for you here. But if you are interested, you can try get it yourself. 
# create an account
# create a project
# select these eight variables: 
# ballot, id, year, hrs1 (hours worked last week), marital, 
# childs, income, happy, 
# (use the search function to find them if needed.)
# add the variables to cart 
# extract data 
# name your extract
# add all the 8 variables to the extract
# Choose output option, select only years 2000 - 2018 
# file format Excel Workbook (data + metadata)
# create extract
# It will take some time to process. 
# When it is ready, click on the download button. 
# you will get a .tar file
# if your system cannot unzip it, google it. (Windows can use 7zip utility. Mac should have it (tar function) built-in.)
# Open in excel (or other comparable software), then save it as csv
# So now you have Happy table to work with
###########################################################################################

################Assignment Starts here#####################################################
##############Read each instruction below very carefully##################################
# Step 1: Preprocessing
# When we import using pandas, we need to do pre-processing 
# So clean up the columns - You can create functions like the total family income, number of children, worked hour last week, etc.
# Some other columns can be manipulated like:
# Ballot: just call it a, b, or c 
# Marital status, it's up to you whether you want to rename the values. 
#You can refer to the accompanying preprocess.py file for better understanding of the dataset and cleaning process. 


#Step 2: Visualization
# After the preprocessing, make these plots..
# Box plot for hours worked last week, for the different marital status. (So x is marital status, and y is hours worked.) 
# Violin plot for income vs happiness, 
# (To use the hue/split option, we need a variable with 2 values/binomial, which 
# we do not have here. So no need to worry about using hue/split for this violinplot.)

#Step 3: Hypothesis Testing (using visualization)
# Use happiness as numeric, make scatterplot with jittering in both x and y between happiness and number of children. Choose what variable you want for hue/color.
# If you have somewhat of a belief that happiness is caused/determined/affected by number of children, or the other 
# way around (having babies/children are caused/determined/affected by happiness), then put the dependent 
# variable in y, and briefly explain your choice.
# %%
assignment_df = pd.read_csv("Happy.csv")

# %%
assignment_df.head(), assignment_df.columns

# %%
# Define functions for data preprocessing

# Clean 'hrs1' (hours worked last week)
def clean_hours_worked(row):
    try:
        hours = int(row["hrs1"])
        return hours if hours >= 0 else np.nan
    except:
        if row["hrs1"].strip() == "Not applicable":
            return np.nan
        return np.nan

# Clean 'childs' (number of children)
def clean_children(row):
    children = str(row["childs"]).strip()
    if children == "Dk na":
        return np.nan
    elif children == "Eight or more":
        return min(8 + np.random.chisquare(2), 12)  # Randomize around 8-12
    else:
        try:
            return int(children)
        except:
            return np.nan

# Clean 'income' (family income ranges)
def clean_income(row):
    income = str(row["income"]).strip()
    if income in ["Don't know", "Not applicable", "Refused", "No answer"]:
        return np.nan
    elif income == "Lt $1000":
        return np.random.uniform(0, 999)
    elif income == "$1000 to 2999":
        return np.random.uniform(1000, 2999)
    elif income == "$3000 to 3999":
        return np.random.uniform(3000, 3999)
    elif income == "$4000 to 4999":
        return np.random.uniform(4000, 4999)
    elif income == "$5000 to 5999":
        return np.random.uniform(5000, 5999)
    elif income == "$6000 to 6999":
        return np.random.uniform(6000, 6999)
    elif income == "$7000 to 7999":
        return np.random.uniform(7000, 7999)
    elif income == "$8000 to 9999":
        return np.random.uniform(8000, 9999)
    elif income == "$10000 - 14999":
        return np.random.uniform(10000, 14999)
    elif income == "$15000 - 19999":
        return np.random.uniform(15000, 19999)
    elif income == "$20000 - 24999":
        return np.random.uniform(20000, 24999)
    elif income == "$25000 or more":
        return 25000 + 10000 * np.random.chisquare(2)  # Apply randomness for higher incomes
    else:
        return np.nan

# Clean 'ballot' (simplify to 'a', 'b', 'c')
def clean_ballot(row):
    return row["ballot"].strip().split()[-1].lower() if pd.notna(row["ballot"]) else np.nan

# Apply the cleaning functions to the dataset
assignment_df["hrs1"] = assignment_df.apply(clean_hours_worked, axis=1)
assignment_df["childs"] = assignment_df.apply(clean_children, axis=1)
assignment_df["income"] = assignment_df.apply(clean_income, axis=1)
assignment_df["ballot"] = assignment_df.apply(clean_ballot, axis=1)

# Check types and initial cleaned data preview
assignment_df.dtypes, assignment_df.head()
