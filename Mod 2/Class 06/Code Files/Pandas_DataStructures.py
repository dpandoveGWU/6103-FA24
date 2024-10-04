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
#%%
#1-D Series 
obj = pd.Series([4, 7, -5, 3])
obj
#%%
#Accessing index and values 
print(obj.array)
print(obj.index)

#%%
#Define all elements of a series
obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
print(obj2)
print(obj2.index)

#%%
#Select single values
print(obj2["a"])
obj2["d"] = 6
print(obj2[["c", "a", "d"]])

#%%
#Operations on a series: Similar to numpy
print(obj2[obj2 > 0])
mul = obj2 * 2
print(mul)
import numpy as np
print(np.exp(obj2))

#%%#
#Series and Dictionaries
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
print("New Series\n",obj3)
print("Original Dictionary\n", obj3.to_dict())

#%%
#Missing Values
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
print("obj 4:",obj4)
#Not the difference between the two functions
print(pd.isna(obj4))
print(pd.notna(obj4))

#%%
#Create DataFrame from a dictionarydata = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
print(frame)
#%%
print("Head\n",frame.head())
print("Tail\n",frame.tail())
#%%
#Create a DataFrame
pd.DataFrame(data, columns=["Year", "State", "Pop"])
frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])
frame2
print(frame2.columns)


#%%
#Retrieve a column 
#Notation 1
print("State:\n")
frame2["state"]
#notation 2
print("Year:\n")
frame2.year
#%%
#Nested Dictionary
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
               "Nevada": {2001: 2.4, 2002: 2.9}}
frame3 = pd.DataFrame(populations)
frame3
#%%
#Transpose a data frame
frame3.T
#%%
#Convert to 2-d numpy array
frame3.to_numpy()
