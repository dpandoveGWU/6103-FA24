#This file contains examples of essential functions in Pandas that help with data analysis and manipulation
#%%
import numpy as np
import pandas as pd
# %%
#Import two common pandas data structures
from pandas import Series, DataFrame
#%%
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
pd.options.display.max_columns = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)

#Reindexing
#An important method on pandas objects is reindex, 
# which means to create a new object with the values rearranged to align with the new index.
#%%
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
obj

#%%
obj2 = obj.reindex(["a", "b", "c", "d", "e"])
obj2

#%%
#Reindex in dataframes work on both rows and columns
frame = pd.DataFrame(np.arange(9).reshape((3, 3)),
                     index=["a", "c", "d"],
                     columns=["Ohio", "Texas", "California"])
print(frame)
frame2 = frame.reindex(index=["a", "b", "c", "d"])
print(frame2)
#%%
#Dropping Entries
obj = pd.Series(np.arange(5.), index=["a", "b", "c", "d", "e"])
print(obj)
new_obj = obj.drop("c")
print("New Object\n")
print(new_obj)
print("More Drops\n:")
obj.drop(["d", "c"])
      
#%%
#Dropping with index and column values 
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=["Ohio", "Colorado", "Utah", "New York"],
                    columns=["one", "two", "three", "four"])
print(data)

#Drop by index
data.drop(index=["Colorado", "Ohio"])

#Drop by column
data.drop(columns=["two"])

#%%
#Indexing, Selection, and Filtering
obj = pd.Series(np.arange(4.), index=["a", "b", "c", "d"])
print("obj:\n", obj)
print("b: ",obj["b"])
print("1: ", obj[1])
print("1,3: \n", obj[[1, 3]])

#%%
#Selection with loc and iloc
#loc:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html#
#Access a group of rows and columns by label(s) or a boolean array.
#iloc:https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html#
#Purely integer-location based indexing for selection by position.
print(obj.loc[["b", "a", "d"]])

#The reason to prefer loc is because of the different treatment of integers 
# when indexing with []. Regular []-based indexing will treat integers as labels 
# if the index contains integers, so the behavior differs depending on the data type of the index. 
# For example:
obj1 = pd.Series([1, 2, 3], index=[2, 0, 1])
print("Obj1: \n",obj1)

obj2 = pd.Series([1, 2, 3], index=["a", "b", "c"])
print("Obj2: \n",obj2)

#%%
#Manipulate obj1 and obj2
obj1 = obj1[[0, 1, 2]]
print("new obj1:\n", obj1)

obj2 = obj2[[0, 1, 2]]
print("new obj2\n:", obj2)

#%%
#iloc works exclustely with integers
print(obj1.iloc[[0, 1, 2]])
print(obj2.iloc[[0, 1, 2]])

#%%
#Try loc on obj2
obj2.loc[[0, 1]]
#Remember loc works exclusively with labels
#%%
#Slicing: Different from normal python as end point is inclusive
obj2.loc["b":"c"]

#%%
#Assigning values
obj2.loc["b":"c"] = 5
print(obj2)

#%%
#Selection on DataFrame with loc and iloc
#Like Series, DataFrame has special attributes loc and iloc for label-based and integer-based indexing, respectively. 
# Since DataFrame is two-dimensional, you can select a subset of the rows and columns with NumPy-like notation using either axis labels (loc) or integers (iloc).
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=["Ohio", "Colorado", "Utah", "New York"],
                    columns=["one", "two", "three", "four"])
data

#%%
print(data.loc["Colorado"])
print(data.loc[["Colorado", "New York"]])

#Select both row and column
print(data.loc["Colorado", ["two", "three"]])

#Similar operation using iloc
print(data.iloc[2],"\n")
print(data.iloc[2, [3, 0, 1]],"\n")
print(data.iloc[[1, 2], [3, 0, 1]],"\n")

#%%
#Index labels with slicing
print(data.loc[:"Utah", "two"])
#Note the format here
print(data.iloc[:, :3][data.three > 5])

#%%
#Boolean arrays can be used with loc but not iloc:
data.loc[data.three >= 2]


#%%
#Arithmetic and Data Alignment
#pandas can make it much simpler to work with objects that have different indexes. 
# For example, when you add objects, if any index pairs are not the same, the respective index in the result will be the union of the index pairs.
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=["a", "c", "d", "e"])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=["a", "c", "e", "f", "g"])
print("s1:\n", s1)
print("s2:\n", s2)
print("Adding both:\n", s1+s2)


#%%
df1 = pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list("bcd"),
                   index=["Ohio", "Texas", "Colorado"])
df2 = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list("bde"),
                   index=["Utah", "Ohio", "Texas", "Oregon"])
print(df1)
print(df2)
print(df1 + df2)
#Think of this as union between two data frames
#%%
#What will happen if there are no common columns or rows between dataframes
df1 = pd.DataFrame({"A": [1, 2,6]})
print(df1)
df2 = pd.DataFrame({"B": [3, 4,6]})
print(df2)
df1 + df2
#%%
#Arithmetic methods with fill values
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),columns=list("abcde"))

df2.loc[1, "b"] = np.nan
print(df1)
print(df2)
print(df1 + df2)

#Fill value
print(df1.add(df2, fill_value=0))

#%%
#Arithmetic operations
print(1 / df1)
print(df1.rdiv(1))
print(df1.reindex(columns=df2.columns, fill_value=0))


#%%
#Sorting and Ranking
#Sorting by index
obj = pd.Series(np.arange(4), index=["d", "a", "b", "c"])
print(obj)
print("sorted obj:\n",obj.sort_index())

# %%
frame = pd.DataFrame(np.arange(8).reshape((2, 4)),
                     index=["three", "one"],
                     columns=["d", "a", "b", "c"])
print(frame)
print("sort by index:\n")
print(frame.sort_index())
print("Sort by col:\n")
print(frame.sort_index(axis="columns"))
#%%
frame.sort_index(axis="columns", ascending=False)
#%%
#Ranking
#Ranking assigns ranks from one through the number of valid data points in an array, 
# starting from the lowest value. 
# The rank methods breaks ties by assigning each group the mean rank
frame = pd.DataFrame({"b": [4.3, 7, -3, 2], "a": [0, 1, 0, 1],
                      "c": [-2, 5, 8, -2.5]})
print(frame)
print(frame.rank(axis="columns"))

#%%
#Summarizing and Computing Descriptive Statistics
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
                   [np.nan, np.nan], [0.75, -1.3]],
                  index=["a", "b", "c", "d"],
                  columns=["one", "two"])
df

#%%
df.sum()

#%%
df.sum(axis="columns")

#%%
#When an entire row or column contains all NA values, the sum is 0, whereas if any value is not NA, then the result is NA. 
# This can be disabled with the skipna option, in which case any NA value in a row or column names the corresponding result NA
df.sum(axis="index", skipna=False)
#%%
df.mean(axis="columns")
#%%
df.idxmax()
#%%
df.cumsum()

#%%
#Multiple summary statistics
df.describe()
# %%
#Unique Values, Value Counts, and Membership
obj = pd.Series(["c", "a", "d", "a", "a", "b", "b", "c", "c"])
# %%
uniques = obj.unique()
uniques
# %%
obj.value_counts()
#%%
#isin performs a vectorized set membership check and can be useful in filtering a dataset down to a subset of values in a Series or column in a DataFrame:
mask = obj.isin(["b", "c"])
mask
# %%
data = pd.DataFrame({"Qu1": [1, 3, 4, 3, 4],
                     "Qu2": [2, 3, 1, 2, 3],
                     "Qu3": [1, 5, 2, 4, 4]})
data

# %%
data["Qu1"].value_counts().sort_index()

#%%
result = data.apply(pd.value_counts).fillna(0)
result
#Research .apply method with regard to dataframes