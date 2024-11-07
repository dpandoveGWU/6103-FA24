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

df = pd.read_csv("Happy.csv")
df.head()

# %%

# Function to clean hours worked last week (hrs1) to ensure numeric values
def worked_hours(df):
    df['hrs1'] = pd.to_numeric(df['hrs1'], errors='coerce')
    return df

# Function to clean number of children (childs) to ensure numeric values
def number_of_children(df):
    df['childs'] = pd.to_numeric(df['childs'], errors='coerce')
    return df

# Function to simplify income ranges to a numeric value for total family income
def family_income(df):
    def income_mapper(value):
        if '8000 to 9999' in value:
            return 9000
        elif '15000 - 19999' in value:
            return 17500
        elif '25000 or more' in value:
            return 25000
        elif value == 'No answer':
            return None
        else:
            return None  # Handles unexpected values
    df['income'] = df['income'].apply(income_mapper)
    return df

# Function to map ballot types to simpler labels (a, b, c)
def ballot(df):
    df['ballot'] = df['ballot'].map({'Ballot a': 'a', 'Ballot b': 'b', 'Ballot c': 'c'})
    return df

# Function to map happiness levels to numeric values
def happiness(df):
    df['happy'] = df['happy'].map({'Not too happy': 1, 'Pretty happy': 2, 'Very happy': 3})
    return df

# Applying each function to clean the respective columns
df = worked_hours(df)
df = number_of_children(df)
df = family_income(df)
df = ballot(df)
df = happiness(df)

# Display the cleaned data to confirm the changes
df.head()


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style of the visualization
sns.set(style="whitegrid")

# Box plot for hours worked last week by marital status
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='marital', y='hrs1')
plt.title('Box Plot of Hours Worked Last Week by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Hours Worked Last Week')
plt.xticks(rotation=45)
plt.show()

# %%
# Violin plot for income vs happiness
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='happy', y='income')
plt.title('Violin Plot of Income vs Happiness')
plt.xlabel('Happiness Level')
plt.ylabel('Income')
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for the plot
sns.set(style="whitegrid")

# Scatter plot with jitter for happiness vs. number of children
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='childs', y='happy', jitter=True, hue='marital', dodge=True)
plt.title('Scatter Plot of Happiness vs Number of Children')
plt.xlabel('Number of Children')
plt.ylabel('Happiness Level')
plt.legend(title='Marital Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# %%
