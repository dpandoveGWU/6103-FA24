# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')

print("\nReady to continue.")

#%%

df = df = pd.read_csv("GSS_demographics.csv")
print("\nReady to continue.")

#%%
# data from gssexplorer, column info:
# ballot	Ballot used for interview
# id	Respondent id number
# year	Gss year for this respondent
# occ	Rs census occupation code (1970)
# sibs	Number of brothers and sisters
# childs	Number of children
# age	Age of respondent
# educ	Highest year of school completed
# paeduc	Highest year school completed, father
# maeduc	Highest year school completed, mother
# speduc	Highest year school completed, spouse
# degree	Rs highest degree
# padeg	Fathers highest degree
# madeg	Mothers highest degree
# spdeg	Spouses highest degree
# sex	Respondents sex
# hompop	Number of persons in household
# income	Total family income
# rincome	Respondents income 

#%%
# Look at one of the columns that should be numeric
print(df.childs.describe())
print(df.childs.value_counts())

print("\nReady to continue.")

#%%
try: 
  df['childs_convert'] = pd.to_numeric( df.childs )
  # try: df.childs = pd.to_numeric( df.childs, errors='coerce' )
  print(df.childs_convert.describe(), '\n', df.childs_convert.value_counts(dropna=False))
except: 
  print("Cannot handle to_numeric for column: childs")
  print(df.childs.describe(), '\n', df.childs.value_counts(dropna=False))
# above doesn't work, since there are many strings there

# Okay, so "Dk na" and "Eight or more" are the bad ones we needa strategy for.
# Say set "Dk na" as NaN, and "Eight or more" as 8, 
# with the understanding that most of them are actually 8, a few 9 or 10 or ... 
print("\nReady to continue.")

#%%
# So try our plan, if it works, put it in the cleaning function, 
# so that if we have newer dataset with the same issues, it will 
# works like a charm
# df.childs.str.replace('Dk na', np.nan ) # default regex=False
df['childs_convert'] = df.childs.map(lambda x: np.nan if str(x).strip() == 'Dk na' else '8' if str(x).strip() == 'Eight or more' else x)
print( df.childs_convert.value_counts(dropna=False) )
print("\nReady to continue.")

#%%
# Alright, it worked. 
# Look one more time, is it int or object?
print(df.dtypes)
# still object, so try again
try: df.childs_convert = pd.to_numeric( df.childs_convert)
except: print("Cannot handle to_numeric for column: childs")
finally: print(df.childs_convert.describe(), '\n', df.childs_convert.value_counts(dropna=False))

# It works, now float64 (because NaN is not int64, it's considered float64, which is okay)
print("\nReady to continue.")

#%%
# Before we move on to clean other columns, this is what I would do: 
# implement cleaning function to complete the task, per row of data 
# What we did above was transforming a single column using the pandas to_numeric( ) funcction. 
# This is not always possible or desireable when df becomes huge. Also, what about you are adding new 
# data points to your df? No reason to apply the transform to the entire column.
# We would like to have a function to perform the cleaning, using .apply( function, axis=1 ) 
# and tranform one row at a time when needed. 
# 
#
# Let us try this on the age column, which is still an object/string
try: df.age = pd.to_numeric( df.age )
except: print("Cannot handle to_numeric for column: age")
finally: print(df.age.describe(), '\n', df.age.value_counts(dropna=False))

print("\nReady to continue.")

#%%
# So let us define our cleaning function to handle the values
def cleanDfAge(row):
  thisage = row["age"]
  try: thisage = int(thisage) # if it is string "36", now int
  except: pass
  
  try: 
    if not isinstance(thisage,int) : thisage = float(thisage)  # no change if already int, or if error when trying
  except: pass
  
  if ( isinstance(thisage,int) or isinstance(thisage,float) ) and not isinstance(thisage, bool): return ( thisage if thisage>=0 else np.nan )
  if isinstance(thisage, bool): return np.nan
  # else: # assume it's string from here onwards
  thisage = thisage.strip()
  if thisage == "No answer": return np.nan
  if thisage == "89 or older": 
    # strategy
    # let us just randomly distribute it, say according to chi-square, 
    # deg of freedom = 2 (see distribution from https://en.wikipedia.org/wiki/Chi-square_distribution for example) 
    # values peak at zero, most range from 0-5 or 6. Let's cap it 100
    thisage = min(89 + 2*np.random.chisquare(2) , 100)
    return thisage # leave it as decimal
  return np.nan # catch all, just in case
# end function cleanGssAge
print("\nReady to continue.")

# For more on np.random distributions, check out the docs
# https://numpy.org/doc/1.16/reference/routines.random.html 
# 


#%%
# Now apply to df row-wise 
# Be mindful of one key point: 
# the .map() function was applied to df.childs or df['childs'], which is a pandas Series itself. 
# The .apply( ) function only works on pandas.DataFrame, not Series. 
# So we need to apply it to df.apply( ), or df[['some subsets']].apply( ), 
# not the single square bracket df[ ].apply( ), which result in a pandas series.
df['age'] = df.apply(cleanDfAge,axis=1)
# df[['age']] = df.apply(cleanDfAge,axis=1) # this works too
print(df.dtypes)
# all works as designed
#
# Next, we can get these individual functions for each column, 
# and in the end, can also combine them all into one dedicated function for this data source
print("\nReady to continue.")

#%%
# Let us take care of one more here: rincome (respondent income), which can be similary applied to family income etc
print(df.rincome.describe(), '\n', df.rincome.value_counts(dropna=False) )
print(df.income.describe(), '\n', df.income.value_counts(dropna=False) )
# Turns out the data is not as meaningful as we had hoped. 
# For rincome (respondent's income), most freq is NA, closely follow by $25k or more. 
# The next is only 1/7 of the frequency for $20k-$25k
# It is clear that the most important info to distinguish those above $25k is not recorded in the dataset. 
# The income (family income) is even more extreme, with no NA but almost twice of those with $25k or more. 
# Nonetheless, as a practice, let us devise a strategy to convert them into slightly more useful values.
# Option 1: spread out each group evenly (but randomized) within the income ranges. 
# Note that the income ranges are not of equal sizes, and the top bin with $25k or more needs some cutoff.
# Option 2: Instead of evenly, we can spread them out using normal pdf. The effect is quite 
# similar to add jittering. We still need to find some reasonable midpoint for the top range.
# Option 3: Replace by mid-points of the range. The top one with $25k or more, 
# we can try something like something like stretching out the age values above 89. 
# But with the majority of the data points in that range, our unsubstantiated choice 
# will affect the result too significantly. 
#
print("\nReady to continue.")

#%%
# I will use Option 1, but smooth out the top bin with chi-square like drop off (deg-of-freedom 2) 
# with the freq from $20-$25k going into $25k+ as smoothly as possible.
def cleanDfIncome(row, colname): # colname can be 'rincome', 'income' etc
    thisamt = str(row[colname]).strip()  # Convert to string to avoid errors
    if thisamt == "Don't know": return np.nan
    if thisamt == "Not applicable": return np.nan
    if thisamt == "Refused": return np.nan 
    if thisamt == "Lt $1000": return np.random.uniform(0, 999)
    if thisamt == "$1000 to 2999": return np.random.uniform(1000, 2999)
    if thisamt == "$3000 to 3999": return np.random.uniform(3000, 3999)
    if thisamt == "$4000 to 4999": return np.random.uniform(4000, 4999)
    if thisamt == "$5000 to 5999": return np.random.uniform(5000, 5999)
    if thisamt == "$6000 to 6999": return np.random.uniform(6000, 6999)
    if thisamt == "$7000 to 7999": return np.random.uniform(7000, 7999)
    if thisamt == "$8000 to 9999": return np.random.uniform(8000, 9999)
    if thisamt == "$10000 - 14999": return np.random.uniform(10000, 14999)
    if thisamt == "$15000 - 19999": return np.random.uniform(15000, 19999)
    if thisamt == "$20000 - 24999": return np.random.uniform(20000, 24999)
    if thisamt == "$25000 or more": return 25000 + 10000 * np.random.chisquare(2)
    return np.nan
# end function cleanDfIncome
print("\nReady to continue.")

#%%
# Now apply to df row-wise. 
# Here with two arguments in the function, we use this syntax
df['income'] = df.apply(cleanDfIncome, colname='income', axis=1)
df.rincome = df.apply(cleanDfIncome, colname='rincome', axis=1)
print(df.dtypes)
# all works as designed
#
print("\nReady to continue.")

#%%
# We are on a roll, let's also rewrite the Childs transform with a function

def cleanDfChilds(row):
  thechildren = row["childs"]
  try: thechildren = int(thechildren) # if it is string "6", now int
  except: pass
  
  try: 
    if not isinstance(thechildren,int) : thechildren = float(thechildren)  # no change if already int, or if error when trying
  except: pass
  
  if ( isinstance(thechildren,int) or isinstance(thechildren,float) ) and not isinstance(thechildren, bool): return ( thechildren if thechildren>=0 else np.nan )
  if isinstance(thechildren, bool): return np.nan
  # else: # assume it's string from here onwards
  thechildren = thechildren.strip()
  if thechildren == "Dk na": return np.nan
  if thechildren == "Eight or more": 
    thechildren = min(8 + np.random.chisquare(2) , 12)
    return thechildren # leave it as decimal
  return np.nan # catch all, just in case
# end function cleanDfChilds

df['childs'] = df.apply(cleanDfChilds, axis=1)
print(df.dtypes)
print("\nReady to continue.")

#%%
# now we can remove our first trial on childs clean up by dropping the column childs_convert
# the lines (or some conbinations) will work. You can try to see how it goes exactly
df = df.drop(columns="childs_convert")
# df = df.drop("childs_convert", axis=1)
# df.drop("childs_convert", axis=1, inplace=True)
print("\nReady to continue.")


#%%
# Now try some plots
df.age.plot.hist()
plt.show()


#%%
# scatter plot
df.plot('age', 'rincome', kind='scatter', marker='o') # OR
# plt.plot(df.age, df.rincome, 'o') # if you put marker='o' here, you will get line plot?
plt.ylabel('Respondent income (annual?)')
plt.xlabel('Age')
plt.show()


#%%
# more adjustments
fuzzyage = df.age + np.random.normal(0,1, size=len(df.age))
#fuzzyrincome = df.rincome + np.random.normal(0,1, size=len(df.rincome))
plt.plot(fuzzyage, df.rincome, 'o', markersize=3, alpha = 0.1)
plt.ylabel('Respondent income (annual?)')
plt.xlabel('Age')
plt.show()



#%%
