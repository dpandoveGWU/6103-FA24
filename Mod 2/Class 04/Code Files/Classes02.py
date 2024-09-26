# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import math
import os

print("Hello world!")

#%%
# Reviews/Summary
# BIFS
# list 
# subsetting
# tuple, set, dictionary 
# loops 
for i in range(0,7) : print(i)
# for index, val in enumerate(list) :
thething = [ 4,'2',("a",5),'end' ]
for val in thething : print (val)
for index, val in enumerate(thething) : print("index", index, ", value", val)
# for looping over dictionaries
thething = { "k0":4, "k8":'2', "k1":("a",5), "k5":'end' }
for key in thething : print("key:", key, ", value:", thething[key] )
# or use thething.items() # creates a object type of dict_items, which can be looped thru as key:value pairs   
for key, val in thething.items() : print("key:", key, ", value:", val)

#%%
# map(function, *iterables)
# the iteration stops when the shortest one reaches the end
# function can be compressed as one-liner, call it "lambda" function as an anonymous function
alist = [1, 2, 5, 3]
blist = [4, 0.5, 1, 1.5, 2, 8]
def dblval(x) : return 2*x
amap = map(dblval, alist)
# the above two lines are the same as the one below, which is useful for small (and non-reuseable) functions
amap = map(lambda x: 2*x, alist)

#%% 
# Be careful. The map object is transient and fleeting ...
amap = map(lambda x: 2*x , alist)
print(amap)
print(list(amap))
print(list(amap))
# notice that the first time you list out amap, it iterated the entire sequence and removed each element
# save it out right away is the common practice
#%%
resultlist = list(map(lambda x: 2*x , alist))
print(resultlist)
print(list(resultlist))
print(list(resultlist)) # still here, still the same

#%% [markdown]
# ## Using the map function 
#
# Try to get a list of numbers given by 1/n^2 using the map() function.
# Next using a loop to add all those numbers from n=1 to 1000 (nmax)
# What is the difference of the value and pi^2 / 6?
# Can you make these into a single function that returns the difference, depending on nmax?
# def findDiff(nmax) :

#%%
nmax = 1000
seq = list( map(lambda n: n**(-2), range(1,nmax+1) ) )
sum = 0
for v in seq: sum = sum+v
print(sum)
print(math.pi**2/6)
print("Difference =", math.pi**2/6-sum)

#%%
def findDiff(nmax):
  seq = list( map(lambda n: n**(-2), range(1,nmax+1) ) )
  sum=0
  for v in seq: sum = sum+v
  return (math.pi**2/6 - sum)

print(findDiff(1000))
print(findDiff(100000000))



#%%
# Now with multiple argument functions
def mypower(x,y): return x ** y 
resultlist = list( map(mypower, alist, blist) )
# or
resultlist = list( map(lambda x,y: x ** y , alist, blist) )
print(resultlist)

#%%
# We could perform the task above ourselves with loops (as a practice)
resultlist = [] # initialize an emtpy list
for i in range(0, min( len(alist),len(blist) ) ):
  resultlist.append( alist[i] ** blist[i] )
print(resultlist)
# Of course when things get more complicated, or if you already have the function defined, it should be much easier to use the map() function.

#%%
# read APPL stock csv file
import os # already imported start of file
print("current directory is : " + os.getcwd())
print("Directory name is : " + os.path.basename(os.getcwd()))
# need to make sure your directory is correct for the file, and use the correct / or \ for your OS/platform

#%%
# filepath = "/Users/edwinlo/GDrive_GWU/github_elo/GWU_classes/DATS_6103_DataMining/Class03_classes/AAPL_20140912_20190912_daily_eod_vol.csv"
appl_date = []
appl_price_eod = []
filepath = os.path.join( os.getcwd(), "AAPL_20140912_20190912_daily_eod_vol.csv")
fh = open(filepath) # fh stands for file handle
# data pulled from https://old.nasdaq.com/symbol/aapl/historical (5 years, csv format, 9/12/2019)   can also try  www.quandl.com/EOD
for aline in fh.readlines(): # readlines creates a list of elements; each element is a line in the txt file, with an ending line return character. 
  tmp = aline.split(',')
  appl_date.append(tmp[0].strip())
  appl_price_eod.append(float(tmp[1]))
  
print(appl_date)
print(appl_price_eod)




#%%
# OOP -- Object Oriented Programming 
# class
# class can be thought of bundling properties (like variables) and functions (or called methods) together
# This is not a requirement, but good practice to use Capitalize first letter for classes, 
# variables or functions instances use regular lowercase first letter.
class Person:
  """ 
  a person with properties and methods 
  height in meteres, weight in kgs
  """

  # contructor and properties
  # __init__ is also called constructor in other propgramming langs
  # it also set the attributes in here 
  def __init__(self, lastname, firstname, height, weight) :
    self.lastname = lastname
    self.firstname = firstname
    self.height_m = height
    self.weight_kg = weight
  
  # find bmi according to CDC formula bmi = weight/(height^2)
  def bmi(self) : 
    return self.weight_kg/(self.height_m ** 2)
  
  def print_info(self) :
    print( self.firstname, self.lastname+"'s height {0:.{digits}f}m, weight {1:.1f}kg, and bmi currently is {2:.{digits}f}".format(self.height_m, self.weight_kg, self.bmi(), digits=2) )
    return None

  # gain weight
  def gain_weight_kg(self,gain) : 
    self.weight_kg = self.weight_kg + gain 
    # return
    return self

  # gain height
  def gain_height_m(self,gain) : 
    self.height_m = self.height_m + gain 
    # return
    return self
  
  def height_in(self) :
    # convert meters to inches
    return self.height_m *100/2.539
  
  def weight_lb(self) :
    # convert meters to inches
    return self.height_m *100/2.539
  
  
  

#%%
# instantiate the Person object as elo, etc
elo = Person('Lo','Edwin',1.6,60)
vars(elo) # shows all attributes and their values
# dir(elo) # shows all attributes and methods

#%%
elo.print_info()
elo.gain_weight_kg(5) # no return value for this method
# same as
# Person.gain_weight_kg(elo,5) # use both arguments here
elo.print_info()

#%%
superman = Person('Man','Super', 1.99, 85)
superman.gain_weight_kg(-3.5)
superman.print_info()

persons = []
persons.append(elo)
persons.append(superman)
print(len(persons))

#%%
# Add to the Person class four other attributes. At least one of the type float or int.
# Add at least three other methods to the class that might be useful


#%% [markdown]
# 
# ## From a programmer's perspective on Object-Oriented Programming (OOP)
# 
# Read this [blog at Medium on OOP](https://medium.com/@cscalfani/goodbye-object-oriented-programming-a59cda4c0e53). 
# To put all these into context, from procedural progamming (such as C) to OOP (C++, java and the likes) was a 
# huge paradigm shift. The world has progressed however, and there are new needs, and new wants, from the new generations. 
# And there are new answers in response. Keep up with the new ideas and concepts. 
# That's how to stay ahead. 
# Just like OOP still uses a lot of concepts and functionality in procedure programming, 
# the new programming paradigm will continue to use OOP concepts and tools as the backbone. 
# Try to get as much as you can, although you might not consider yourself a programmer. 
# These will serve you well, and makes you a more logical thinker.


#%%
# Do it together 

# class Cars :
  
# %%




