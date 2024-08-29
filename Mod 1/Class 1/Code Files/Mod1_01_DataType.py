#%%[markdown]
# # PYTHON IS SENSITIVE to INDENTATIONS
# Rmember this if nothing else about python!! 
# 
# We will get into that very soon once we go beyond the HelloWorld type codes.


#%%[markdown]
#
# # Python Mod 02
# 
# Above was an h1 header.
# Hello to everyone.   
#
# This can get you a [link](http://www.gwu.edu).
#
# You can find some cheatsheets to do other basic stuff like bold-face, italicize, tables, etc.

#%%
# # Four basic data type in Python (compared to Java/C++, python is very limited that way, for good and bad)
# BIFS - boolean, integer, float, string

abool = True    # boolean
aboolstr = "True"     # str
azero = 0    # int
aint = 35    # int
aintstr = "35"     # str
afloat = -2.8     # float
afloatstr = "-2.8"     # str
anumbzerostr = "0"     # str
aemptystr = ""     # str
aletter = 'a'     # str
astr = "three-five"     # str

# %%
# First, let us try a little interactive - allow user input to set a parameter
userin = input("What is your name?")
print(f'Hello {userin}\n')

# %%
userin = input("What is your favorite integer?")
print(f'Your fav: {userin}')
print(f'Your fav: doubled: {userin * 2}\n')

# %%
# TRY AGAIN
userin = int(input("What is your favorite integer?"))
print(f'Your fav: {userin}')
print(f'Your fav: doubled: {userin * 2}\n')

# OR 
# Look up the eval() function in python, see what it does. 
userin = eval(input("What is your favorite integer?"))
print(f'Your fav: {userin}')
print(f'Your fav: doubled: {userin * 2}\n')

# f string inside the print function 


#%%
# higher level data types (class)
# list / array
alist = [1,'person',1,'heart',10,'fingers']

# tuple # like list, but an immutable object (faster access and processing)
atuple = (1,'person',1,'heart',10,'fingers')

# set  # un-ordered, and distinct elements.  
aset = { 1,'person',1,'heart',10,'fingers' }

#%%
# Access elements in a list:
# Remember that python COUNTS starting FROM ZERO
print(f'The list alist has length {len(alist)}')
print(f'Recall this is alist: {alist}')
print(f'This is alist[0]: {alist[0]}')
print(f'This is alist[1]: {alist[1]}')
print(f'This is alist[5]: {alist[5]}')
print(f'This is alist[-1]: {alist[-1]}')
print(f'This is alist[-4]: {alist[-5]}') # should be same as alist[1], or in general,  the positive number minus the length
print(f'This is alist[6]: {alist[6]}') # ERROR
#
#%%
# Same rules for tuples.
# 
# The difference? You an also use this to assign new values to an element in a list.
alist[2] = 2
alist[3] = 'hands'
print(f'This is the new alist: {alist}')
#
# But you will get an error for atuple
atuple[2] = 2
# That's what it means by tuple is "immutable"
# TypeError: 'tuple' object does not support item assignment
# 
#%%
# Sets, on the other hand, are not ordered. Parts of a set cannot be retrived like that. 
aset[1]
# TypeError: 'set' object is not subscriptable 


#%%
# dictionary # like associative array in other lang.  
# The list is not indexed (by integers), but reference by a key.
# #######################################
# The key must be a primitive data type 
# preferrably use only Int and Str !!!
# #######################################
adictionary = { "name": "Einstein", 1: "one", "love": 3.14159 }
# access elements with 
adictionary['love']

#%%
# This kind of works too?! 
# Also kind of strange to use float and bool as key though, but it is possible and you might find it reasonable in certain situations.
adictionary2 = { "name": "Einstein", 1: "one", 3.14: 3.14159, True: 'love', "last": alist }
print(adictionary2)
print(type(adictionary2["last"]))
print(len(adictionary2))

#%%
# ######## BUT BE VERY CAREFUL if you use bool and float for keys. They might not be what you expect.
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }
print(adictionary3)
print(len(adictionary3))
adictionary4 = { "name": "Einstein", 2: "two", 2.0: 3.14159, True: 'loves', "last": alist }
print(adictionary3)
print(len(adictionary3))
# below does not work. you can try by uncommenting it and run the line code
# notadicationary = { ['a',2]: "sorry", {1,2}: ('not','ok') }

#%%
# ###################  1. Exercise    Exercise    Exercise   ################################## 
# Try to create some more complicated entities. List of tuples, dictionary of dictionary, see if you find anything unexpected. 
print("This is exercise 1")




#%%
# ###################  2. Exercise    Exercise    Exercise   ################################## 
# Implicit conversion, which is also called coercion, is automatically done. (different lang has different coercion rules.)
# Explicit conversion, which is also called casting, is performed by code instructions.
print("This is exercise 2")


#%%
# Example, try 
print(int(abool))
print(str(abool))
print(str(int(abool)))
# int(str(abool))

#%%
# Try it yourself, using the functions bool(), int(), float(), str() to convert. 
# what are the ones that you surprises you? List them out for your own sake




#%% 
# ####################  3. Exercise - binary operations:  ################################## 
# try to add or multiply differnt combinations and see the result. 
# Show your work here
print("This is exercise 3")

# Example -- implicit conversion is automatic here
add_bool_zero = abool + azero
print('result: type= ' + str(type(add_bool_zero)) + ' , value= ' +str(add_bool_zero) )



#%%
# ####################  4. Exercise - copying/cloning/deep cloning/shallow copy  ################################## 
# copy vs reference 
print("This is exercise 4")
abool = True    # boolean
cbool = abool   # make a copy
print(f'abool = {abool}')
print(f'cbool = {cbool}')
# To check if two things are equal 
print(f'??cbool == abool?? : {cbool == abool}') # element-wise check
print(f'??cbool is abool?? : {cbool is abool}') 

print('Now change one, and check')
abool = False
print(f'abool = {abool}')
print(f'cbool = {cbool}')
print(f'??cbool == abool?? : {cbool == abool}') # element-wise check
print(f'??cbool is abool?? : {cbool is abool}') 
#do the same for the four differnt types
#


#%%
# ####################  Next, try it on tuple, list, set, dictionary ####################
ctuple = atuple
print(f'??ctuple == atuple?? : {ctuple == atuple}') # element-wise check
print(f'??ctuple is atuple?? : {ctuple is atuple}') 
#
print('Now try to change one')
# ctuple[1] = 'people'  # can't do this. tuple is immutable
# At least we can do this:
ctuple = (1,'person','2','hearts', 6 , 'fingers') # This re-assign the variable ctuple. They are no longer the same.
print(f'atuple = {atuple}')
print(f'ctuple = {ctuple}')
print(f'??ctuple == atuple?? : {ctuple == atuple}') # element-wise check
print(f'??ctuple is atuple?? : {ctuple is atuple}') 
# notice that tuples cannot assign a new value individually like atuple[1]='guy', but you can reassign the entire variable



#%%
# ####################  Next, try it on list, set, dictionary ####################
alist = [1,'person',1,'heart',10,'fingers'] # reset alist
clist = alist # make copy of alist
print(f'??clist == alist?? : {clist == alist}') # element-wise check
print(f'??clist is alist?? : {clist is alist}') # memory address check
#
print('Now try to change one like \n clist[1] = "people"')
clist[0] = 2
clist[1] = 'people'
print(f'alist = {alist}')
print(f'clist = {clist}')
# Both changed!! Is it what you expect??
# To check
print(f'??clist == alist?? : {clist == alist}') # element-wise check
print(f'??clist is alist?? : {clist is alist}') # memory address check
# They are shallow copies



#%%
# HOW to make true copy or deep copy???
# For simple (not nested) list/tuple/dictiionary, we can use these:
# 
alist = [1,'person',1,'heart',10,'fingers'] # reset alist
# the three lines below work the same
clist = list(alist)
clist = alist[:]
clist = alist.copy()
# clist = alist # this line is different, shallow copy, only pointing to the same memory locations
# At this point, even though alist and clist have all the exact same elements, we find:
print("Both lists have same elements:")
print(f'alist = {alist}')
print(f'clist = {clist}')
print(f'??alist == clist?? : {alist == clist}') # element-wise check
print(f'??alist is clist?? : {alist is clist}') # memory address check
# 
print("Now change clist:")
clist[2]=2
clist[3] = 'hands'
print(f'alist = {alist}')
print(f'clist = {clist}')
print(f'??alist == clist?? : {alist == clist}') # element-wise check
print(f'??alist is clist?? : {alist is clist}') # memory address check

#%%
# Now try the other data types: set, dictionary, set of dictionaries, list of tuples, 
# etc etc
# These are shallow copies. They just copy the reference address, not the (primitive) values. 
# How do we make static clones that are no longer tied?
# Try google
# Does that work for deep level objects like list of dictionaries?
#
print(alist) # check the values for alist
# reset adictionary3
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }
# len(adictionary3)
acopy1 = adictionary3
acopy2 = adictionary3.copy()

import copy
acopy3 = copy.copy(adictionary3)

acopy1[2] = 'duo'

print(adictionary3)
print("acopy1", acopy1)
print("acopy2",acopy2)
print("acopy3",acopy3)

 
#%%
# Let us get some help from the package "copy"
#
# reset alist
alist = [1, 'person', 1, 'heart', 10, 'fingers']
# reset adictionary3
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }

# len(adictionary3)
acopy1 = adictionary3
acopy2 = adictionary3.copy()

#%%
import copy
acopy3 = copy.copy(adictionary3)
acopy4 = copy.deepcopy(adictionary3)

acopy1[2] = 'duo'
acopy1['last'][3]='nose'

print(adictionary3)
print("acopy1",acopy1)
print("acopy2",acopy2)
print("acopy3",acopy3)
print("acopy4",acopy4)


# %%[markdown]
# The copy.deepcopy() method works! It works recursively on lists/dictionary, etc, such as JSON objects. 
# Needless to say, use it only if it is needed, as it costs performance-wise. It also might not work if 
# the object type is other more complicated objects.
#


#%%
# From before
abool = True    # boolean
azero = 0    # int
aint = 35    # int
afloat = -2.8     # float
anumbzerostr = "0"     # str
aemptystr = ""     # str
aletter = 'a'     # str
astr = "three-five"     # str
# list / array
alist = [1,'person',1,'heart',10,'fingers']
# tuple # like list, but immutable (faster access and processing)
atuple = (1,'person',1,'heart',10,'fingers')
# set  # un-ordered, and distinct elements.  
aset = { 1,'person',1,'heart',10,'fingers' }
# dictionary
adictionary = { "name": "Einstein", 1: "one", astr: 35, aint: 'thirty five', "last": alist }

#%%
# some more 
# note anything unexpected/unusual
list1 = [1,5,3,8,2]
list2 = [2]
tuple1 = (1,5,3,8,2)
print("type of tuple1: %s, length of tuple1: %d" % (type(tuple1), len(tuple1)) )

tuple2 = (2)
print("type of tuple2: %s" % type(tuple2) )
# print("type of tuple2: %s, length of tuple2: %d" % (type(tuple2), len(tuple2)) )
# len(tuple2) # does not work, error

tuple3 = tuple([2])
print("type of tuple3: %s, length of tuple3: %d" % (type(tuple3), len(tuple3)) )

tuple4 = ()
print("type of tuple4: %s, length of tuple4: %d" % (type(tuple4), len(tuple4)) )


#%%
# Slicing parts of list/tuple/set
# Try
# write some notes/comments for each case, so that you can review them easily yourself
alist[1:4]  # inclusive on the start index, exclusive of the end index
alist[:4]
alist[:]
# optional argument, skipping every 1 element with :2 at the end
alist[1:4:2]
alist[1:5:2]
alist[1:3:2]
# what do you expect the result of this to be?
alist[1::2]
#%%
# Also try 
alist[-4]
alist[-4:-2]
alist[-4:]
alist[-2:-4]

#%%
# Now try tuple, set, and dictionary
# Put some notes for yourself
# comment out the illegal ones so that you can run your entire file gracefully

#%%
