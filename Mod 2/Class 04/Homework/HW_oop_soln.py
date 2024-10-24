# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# # HW OOP
# ## By: xxx
# ### Date: xxxxxxx
#

#%% [markdown]
# # Part I
# 
# Remember toggle between line wrap, (Option-Z on Mac, and Alt-Z on Windows?) to view 
# longer lines. I like to keep some lines intact, to keep the overall structure easier 
# to follow.
# 
# Let us try to use OOP to manage something useful.
# 
# This first part of the exercise, we will use just basic python and OOP to 
# create a Stock class, to keep track of different stocks in a portfolio. 
# Let's say, Apple. (NO numpy nor pandas here in this Part I.)
# 
# I pulled data from https://old.nasdaq.com/symbol/aapl/historical (5 years, 
# csv format, 9/12/2019). You can also try from something like  www.quandl.com/EOD
# 
# The csv file is given to you here, along with Microsoft (MSFT) and Google (the parent 
# company is Alphabet now, stock symbol GOOG) files. 
#
# Our goal: Pull in the basic data, store it into a Stock class object. 
# Next, derive some useful numbers and functions out of it.
# For example, from the daily data with only date, end-of-day closing price, and 
# trading volumn as three columns, we want to be able to find the daily changes easily, 
# and also the change of the last 50 days, 100 days, for example. 
# The daily change represents the "first derivatives" in calculus or economics term. 
# It is also very common to look at the "second derivatives", which is the "rate of 
# change of the rate of change of the stock price". This "second derivative" represent 
# something like a overall trend/momentum. A negative 2nd derivative means even though 
# the stock price change (1st derivative) is positive, but the increase is slowing down. 
# You might want to find a time to get out.
# 
# So this is the structure of my proposed Stock class:
# Attributes: 
#   :param symbol: stock symbol
#   :param name: company name
#   :param firstdate: the first date (end of list) of the price list
#   :param lastdate: the last date (beginning of list) of the price list
#   :param init_filepath: locate the file for date, price (eod) and volume lists to initialize this stock object
# below is other attributes that will be kept as a python list. When the object is initialized, 
# it reads in the data file and calculate the rest
#   :self.dates: = [] # starts from the latest/newest date, for easy access of newer data when the list gets long
#   :self.price_eod: = [] # import from data file
#   :self.volumes: = [] # import from data file
#   :self.delta1: = [] # daily change values, or the first derivative, that is the previous close minus today's close, example price_eod[0] - price_eod[1], price_eod[1] - price_eod[2]
#   :self.delta2: = [] # the second derivative, calculate from teh first derivative. Basicallly, the daily change in the first derivative (over 1 day), is the second derivative. 
# 
# Methods:
#   import_history      # import from csv file, then populate the corresponding values for the attributes
#   compute_delta1_list # from the price_eod (a python list), calculate a list of daily changes. Notice that if your eod list has length n, you will only get a list of length n-1 for this list.
#   compute_delta2_list # similarly calculated from delta1 list. And if your delta1 list has length n-1, you will only get a list of length n-2 here.
#   insert_newday       # Every day, I will need to insert a new row of data to keep up-to-date. This task involves adding values to eod, dates, and volumn lists, and also need to calculate and add new values to delta1 and delta2.
#   nday_change_percent # Need to calculate the percent change of price in the last n-days.  
# 
# 
# Most of the codes are written for you. Your task is to understand them, and fill in the rest 
# to make it work.
# 
# Let's start: 

#%%
# Step 0, try reading the data file to make sure it works. If your data file is in the same 
# folder, you don't need to put in the full path. Just the file name. 
#
# Run this cell as is. If you see some output in the interactive python window, then it is 
# working. If not, you might need to fix the file path accordingly for your OS/platform.
# You can use functions like os.getcwd() to find out what is the current working directory 
# in your ipython session. If needed, you can use os.chdir( your path ) to set the current 
# directory to where you need. like:
# filepath = "/Users/edwinlo/GDrive_GWU/github_elo/GWU_classes_p/DATS_6103_DataMining/Assignment/AAPL_daily.csv"
import os
appl_date = []
appl_price_eod = []
filepath = os.path.join( os.getcwd(), "AAPL_daily.csv")
fh = open(filepath) # fh stands for file handle
# data pulled from https://old.nasdaq.com/symbol/aapl/historical (5 years, csv format, 9/12/2019)   can also try  www.quandl.com/EOD
for aline in fh.readlines(): # readlines creates a list of elements; each element is a line in the txt file, with an ending line return character. 
  # this file gives "23.57" as the string, including the quotes
  tmp = aline.split(',')
  appl_date.append(tmp[0].strip())
  appl_price_eod.append(float(tmp[1]))
  
print(appl_date)
print(appl_price_eod)

# You should be able to see the lists created from the code above. This is just our test run.

#%% 
# Step 1
# Create a class for a stock with daily end-of-day price recorded, along with the daily volume.
# 

class Stock:
  """
  Stock class of a publicly traded stock on a major market
  """
  def __init__(self, symbol, name, firstdate, lastdate, init_filepath) :
    """
    :param symbol: stock symbol
    :param name: company name
    :param firstdate: the first date (end of list) of the price list
    :param lastdate: the last date (beginning of list) of the price list
    :param init_filepath: locate the file for date, price (eod) and volume lists
    """
    # note that the complete list of properties/attributes below has more than items than 
    # the numnber of arguments of the constructor. That's perfectly fine. 
    # Some property values are to be assigned later after instantiation.
    self.symbol = symbol.upper()
    self.name = name
    self.firstdate = firstdate
    self.lastdate = lastdate
    # below can be started with empty lists, then read in data file and calculate the rest
    self.price_eod = []
    self.volumes = []
    self.dates = [] # starts from the latest/newest date, 
    self.delta1 = [] # daily change values, the previous close minus today's close, example eod[0] - eod[1], eod[1] - eod[2]
    self.delta2 = [] 
    # change of the daily change values (second derivative, acceleration), 
    # given by, for the first entry, (delta1[0] - delta[1]), 
    # or if we want to, equals to (eod[0]-eod[1]) - (eod[1]-eod[2]) = eod[0] - 2*eod[1] + eod[2]
    self.import_history(init_filepath)
    self.compute_delta1_list() # Calculate the daily change values from stock price itself.
    self.compute_delta2_list() # Calculate the daily values of whether the increase or decrease of the stock price is accelerating. A.k.a. the second derivative.
  # END of constructor/__init__
  
  def import_history(self, filepath):
    """
    import stock history from csv file, with colunms date, eod_price, volume, and save them to the lists 
    """
    with open(filepath,'r') as fh: # leaving the filehandle inside the "with" clause will close it properly when done. Otherwise, remember to close it when finished
      for aline in fh.readlines(): # readlines creates a list of elements; each element is a line in the txt file, with an ending line return character. 
        tmp = aline.split(',')
        self.dates.append(tmp[0].strip())
        self.price_eod.append(float(tmp[1]))
        self.volumes.append(float(tmp[2]))
        tmp = None # reset the dummy variable just in case
    # fh.close() # close the file handle when done if it was not inside the "with" clause
    # print('fh closed:',fh.closed) # will print out confirmation  fh closed: True
    return self
  
  def compute_delta1_list(self):
    """
    compute the daily change for the entire list of price_eod 
    """
    # eod_shift1 = self.price_eod # THIS WILL NOT WORK. Try. A shallow copy.
    eod_shift1 = self.price_eod.copy() # if you do not use the copy method here, you will get a shallow copy.
    # The list here is a simple list of floats, not list of lists or list of dictionaries. 
    # So the copy() function will work. No need for other "deepcopy" variations
    eod_shift1.pop(0) # remove the first element (shifting the day)
    self.delta1 = list(map(lambda x,y: x-y, self.price_eod, eod_shift1))
    print(self.name.upper(),": The latest 5 daily changes: ")
    for i in range(0,5): print(self.delta1[i]) # checking the first five values
    return self
  
  def compute_delta2_list(self):
    """
    compute the daily change for the entire list of delta1, essentially the second derivatives for price_eod
    """
    # essentially the same function as compute_delta1_list. With some hindsight, or when the codes are re-factored, we can properly combine them
    delta1_shift1 = self.delta1.copy() 
    delta1_shift1.pop(0) # remove the first element (shifting the day)
    self.delta2 = list(map(lambda x,y: x-y, self.delta1, delta1_shift1))
    print(self.name.upper(),": The latest 5 2nd-derivatives are: ")
    for i in range(0,5): print(self.delta2[i]) # checking the first five values
    return self
  
  def insert_newday(self, newdate, newprice, newvolume):
    """
    add a new data point at the beginning of lists
    """
    # Make plans (and placeholders)
    # insert newdate to dates[]
    self.dates.insert(0,newdate)
    # insert newvolume to volumes[]
    self.volumes.insert(0,newvolume)
    # insert new price data to price_eod
    self.price_eod.insert(0, newprice)
    # calculate and insert new data to delta1
    self.delta1.insert(0, newprice - self.price_eod[1])
    # calculate and insert new data to delta2
    self.delta2.insert(0, self.delta1[0]-self.delta1[1])
    #
    return self
  
  def nday_change_percent(self,n):
    """
    calculate the percentage change in the last n days, returning a percentage between 0 and 100, or sometimes higher.
    """
    change = self.price_eod[0]-self.price_eod[n]
    percent = 100*change/self.price_eod[n]
    print(f"{self.symbol} : Percent change in {n} days is {percent.__round__(2)}%")
    return percent
    
  

#%%
import os

# dirpath = os.getcwd() # print("current directory is : " + dirpath)
# using os.path.join will take care of difference between 
# mac/pc/platform issues how folder paths are used, backslash/forward-slash/etc
filepath = os.path.join( os.getcwd(), 'AAPL_daily.csv')
# or just this should work
# filepath = 'AAPL_daily.csv'
aapl = Stock('AAPL','Apple Inc','9/12/14','9/12/19',filepath) # aapl is instantiated!

#%%
# Great! Now we can get the competitors easily
filepath = 'MSFT_daily.csv'
msft = Stock('MSFT','Microsoft Inc','9/12/14','9/12/19',filepath)

filepath = 'GOOG_daily.csv'
goog = Stock('GOOG','Alphabet Inc','9/12/14','9/12/19',filepath)



#%%
# Which stock (out of the three) perform best in the last (i) 50 days, (ii) 200 days, (iii) 600 days?
# (i) 50 days
aapl.nday_change_percent(50)
msft.nday_change_percent(50)
goog.nday_change_percent(50)
# GOOG wins

# (ii) 200 days (about 1 year)
aapl.nday_change_percent(200)
msft.nday_change_percent(200)
goog.nday_change_percent(200)
# MSFT wins

# (iii) 600 days (about 3 years)
aapl.nday_change_percent(600)
msft.nday_change_percent(600)
goog.nday_change_percent(600)
# MSFT wins again

# AAPL : Percent change in 50 days is 10.04
# MSFT : Percent change in 50 days is 0.69
# GOOG : Percent change in 50 days is 11.07
# AAPL : Percent change in 200 days is 29.48
# MSFT : Percent change in 200 days is 33.42
# GOOG : Percent change in 200 days is 20.55
# AAPL : Percent change in 600 days is 54.35
# MSFT : Percent change in 600 days is 102.47
# GOOG : Percent change in 600 days is 41.49

#%%
# Now see if the insert_newday() method works
aapl.insert_newday('9/13/19',231.85,32571922)
print('new dates:',aapl.dates[0:5])
print('new price_eod:',aapl.price_eod[0:5])
print('new volumes:',aapl.volumes[0:5])
print('new delta1:',aapl.delta1[0:5])
print('new delta2:',aapl.delta2[0:5])
print('last two days change: ',aapl.nday_change_percent(2))

# new dates: ['9/13/19', '9/12/19', '9/11/19', '9/10/19', '9/9/19']
# new price_eod: [231.85, 223.085, 223.59, 216.7, 214.17]
# new volumes: [32571922, 32226670.0, 44289650.0, 31777930.0, 27309400.0]
# new delta1: [8.764999999999986, -0.5049999999999955, 6.890000000000015, 2.530000000000001, 0.9099999999999966]
# new delta2: [9.269999999999982, -7.39500000000001, 4.360000000000014, 1.6200000000000045, 0.9300000000000068]
# AAPL : Percent change in 2 days is 3.69
# last two days change:  3.694261818507085
#%%
# # Part 2 
# This exercise to be re-done after we learn enough on Pandas. 
# The features in Numpy and Pandas makes a lot of things easier. 

