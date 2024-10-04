# %%
import pandas as pd


# %%
# Importing csv files
hprice = pd.read_csv("hprice1.csv")
hprice


# %%
# Take a peak
hprice.head()


# %%
# Loses the column names
hprice = pd.read_csv("hprice1.csv", header = None)
hprice.head()


# %%
pd.read_csv("hprice1.csv", nrows = 4)


# %%
# Try the following commands
hprice.head()
hprice.tail()
hprice.info()


# %%
# If you did some data processing, and like to save the result as a csv file, 
# you can `import os`
import os
os.getcwd()  # make sure you know what folder you are in

# %%
os.chdir('.') # do whatever you need to get to the right folder, 
# or make sure you include the full path when saving the file
hprice.to_csv('hprice_clean.csv')  

# %%
