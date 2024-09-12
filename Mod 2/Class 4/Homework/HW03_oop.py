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
    self.price_eod = [] # record the end-of-day prices of the stock in a list. The 0-th position is the latest end-of-day price
    self.volumes = [] # a list recording the daily trading volumn
    self.dates = [] # starts from the latest/newest date, 
    self.delta1 = [] # daily change values, today's close price minus the previous close price. Example eod[0] - eod[1], eod[1] - eod[2], 
    self.delta2 = [] # daily change values, the previous close minus today's close, example eod[0] - eod[1], eod[1] - eod[2]
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

        #  ######   QUESTION 1    ######   QUESTION 1    ######   QUESTION 1    ######   QUESTION 1    ######  
        # Fill in the codes here to put the right info in the lists self.dates, self.price_eod, self.volumes  
        # Should be similar to the codes in Step 0 above. 
        #  ######  END QUESTION 1 ######  END QUESTION 1 ######  END QUESTION 1 ######  END QUESTION 1 ######  


    # fh.close() # close the file handle when done if it was not inside the "with" clause
    # print('fh closed:',fh.closed) # will print out confirmation  fh closed: True
    return self
  
  def compute_delta1_list(self):
    """
    compute the daily change for the entire list of price_eod 
    """
    # goal: calculate the daily price change from the eod prices.
    # idea: 
    # 1. duplicate the eod list 
    # 2. shift this new list by removing the 0-th element. 
    # 3. use the map function to find a list of delta's by subtracting the eod list from this new list. 
    # Okay, let's try
    #
    # eod_shift1 = self.price_eod # THIS WILL NOT WORK. Try. A shallow copy. We'll talk more about that next class.
    eod_shift1 = self.price_eod.copy() # if you do not use the copy method here, you will get a shallow copy.
    # The list here is a simple list of floats, not list of lists or list of dictionaries. 
    # So the copy() function will work. No need for other "deepcopy" variations
    eod_shift1.pop(0) # remove the first element (shifting the day)
    self.delta1 = list(map(lambda x,y: x-y, self.price_eod, eod_shift1))
    print(self.name.upper(),": The latest 5 daily changes in delta1: ")
    for i in range(0,5): print(self.delta1[i]) # checking the first five values
    return self
  
  def compute_delta2_list(self):
    """
    compute the daily change for the entire list of delta1, essentially the second derivatives for price_eod
    """
    # essentially the same function as compute_delta1_list. With some hindsight, or when the codes are re-factored, we can properly combine them

    #  ######   QUESTION 2    ######   QUESTION 2    ######   QUESTION 2    ######   QUESTION 2    ######  
    # Fill in the codes here 
    # Need to find the daily changes of the daily change, and save it to the list self.delta2
    # It is the second derivative, the acceleration (or deceleration if negative) of the stock momentum.
    # Essentially the same as compute_delta1_list, just on a different list 
    # Again you might want to print out the first few values of the delta2 list to inspect
    #  ######  END QUESTION 2 ######  END QUESTION 2 ######  END QUESTION 2 ######  END QUESTION 2 ######  

    return self
  
  def insert_newday(self, newdate, newprice, newvolume):
    """
    add a new data point at the beginning of lists
    """
    #  ######   QUESTION 3    ######   QUESTION 3    ######   QUESTION 3    ######   QUESTION 3    ######  
    # After we have the batch of historical data to import, we 
    # most likely will need to do some daily updates (cron jobs, for example) 
    # going forward.  There is no need to re-import the old data. 
    # This method is then used to insert just one row of new data point daily. 
    # We will need to insert the new date, the new eod value, the new delta1 value, 
    # the new delta2 value, as well as the new volume data.
    #
    # insert new price data to price_eod
    # calculate and insert new data to delta1
    # calculate and insert new data to delta2
    # insert newdate to dates[]
    #
    # Fill in the codes here 
    #
    # insert newdate to dates[]
    self.dates.insert('Something Here')
    # insert newvolume to volumes[]
    self.volumes.insert('Something Here')
    # insert new eod data value to price_eod
    self.price_eod.insert('Something Here')
    # calculate and insert new data to delta1
    self.delta1.insert('Something Here')
    # calculate and insert new data to delta2
    self.delta2.insert('Something Here')
    #
    #  ######  END QUESTION 3 ######  END QUESTION 3 ######  END QUESTION 3 ######  END QUESTION 3 ######  

    return self
  
  def nday_change_percent(self,n):
    """
    calculate the percentage change in the last n days, returning a percentage between 0 and 100, or sometimes higher.
      """
    #  ######   QUESTION 4    ######   QUESTION 4    ######   QUESTION 4    ######   QUESTION 4    ######  
    change = 'What should it be?' # calculate the change of price between newest price and n days ago
    percent = 'What should it be?' # calculate the percent change (using the price n days ago as the base)
    print(f"{self.symbol} : Percent change in {n} days is {percent.__round__(2)}%")
    #  ######  END QUESTION 4 ######  END QUESTION 4 ######  END QUESTION 4 ######  END QUESTION 4 ######  

    return percent
    
  

#%%
import os

# dirpath = os.getcwd() # print("current directory is : " + dirpath)
# filepath = dirpath+'/AAPL_daily.csv' # lastdate is 9/12/19, firstdate is 9/12/14, 
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
#  ######   QUESTION 6    ######   QUESTION 6    ######   QUESTION 6    ######   QUESTION 6    ######  
#
# use the nday_change_percent method that you defined
# Find out the stock performances in their percent changes in the last 
# (i) 50 days
# (ii) 200 days (about 1 year)
# (iii) 600 days (about 3 years)
# Which one perform best in each of the periods above?? 
# 
#  ######  END QUESTION 6 ######  END QUESTION 6 ######  END QUESTION 6 ######  END QUESTION 6 ######  


#%%
# 
#  ######   QUESTION 7    ######   QUESTION 7    ######   QUESTION 7    ######   QUESTION 7    ######  
#
# Now see if the insert_newday() method works
aapl.insert_newday('9/13/19',231.85,32571922)
print('new dates:',aapl.dates[0:5])
print('new price_eod:',aapl.price_eod[0:5])
print('new volumes:',aapl.volumes[0:5])
print('new delta1:',aapl.delta1[0:5])
print('new delta2:',aapl.delta2[0:5])
print('last two days change: ',aapl.nday_change_percent(2))
# If the above printouts does not look right to you, make sure you fix your 
# insert_newday function!!
#

#  ######  END QUESTION 7 ######  END QUESTION 7 ######  END QUESTION 7 ######  END QUESTION 7 ######  


#%%
# # Part 2 
# This exercise to be re-done after we learn enough on Pandas. 
# The features in Numpy and Pandas makes a lot of things easier. 
#
#%%