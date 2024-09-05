# BIFS - boolean, integer, float, string

# %%
# More on strings
mesg = 'Hello, world!'
print( mesg[0] )

# How to print the trailing exclamation?
# print( what goes here? )

# Print the length of a string.
# len(mesg)
# How to print the last character of a string of unknown length
# your code here


# %%
# Can we replace a character in a string?
# mesg = "Hollo, Class!"
# mesg[1] = 'e'
# Strings are not mutable in Python

# %%
# Activity - 3
# Explain what the line following lines of code are doing. 
a = 'Apple'
b = a
a = a + a

print(a)
print(b)



# %%
# higher level data types (class)
# list / array
alist = [1, 'person', 1, 'heart', 10, 'fingers']

# tuple: like a list, but an immutable object (faster access and processing)
atuple = (1, 'person', 1, 'heart', 10, 'fingers')

# set: un-ordered, and distinct elements.  
aset = { 1, 'person', 1, 'heart', 10, 'fingers' }


# %%
# dictionary 
# like associative array in other lang.  
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
# Also kind of strange to use float and bool as key though, but it is possible 
# and you might find it reasonable in certain situations.
adictionary2 = { "name": "Einstein", 1: "one", 3.14: 3.14159, True: 'love', "last": alist }
print(adictionary2)
print(type(adictionary2["last"]))
print(len(adictionary2))


#%%
# ######## BUT BE VERY CAREFUL if you use bool and float for keys. 
# They might not be what you expect.
adictionary3 = { "name": "Einstein", 2: "two", 3.14: 3.14159, True: 'loves', "last": alist }
print(adictionary3)
print(len(adictionary3))
adictionary4 = { "name": "Einstein", 2: "two", 2.0: 3.14159, True: 'loves', "last": alist }
print(adictionary3)
print(len(adictionary3))

# below does not work. you can try by uncommenting it and run the line code
# notadicationary = { ['a',2]: "sorry", {1,2}: ('not','ok') }


#%%
# Examples of casting in Python
print(int(True))
print(str(True))
print(str(int(0.02)))


#%%
# ####################  Next, try it on list, set, dictionary ####################
clist = alist
clist[1] = 'people'
print(alist)
print(clist)

#%%
clist = list(alist)
#clist = alist[:]
clist[2]=2
clist[3] = 'hands'
print(alist)
print(clist)
# Is it what you expect??

#%%
# Now try the other data types: set, dictionary, set of dictionaries, list of tuples, 
# etc etc
# These are shallow copies. They just copy the reference address, not the (primitive) values. 
# How do we make static clones that are no longer tied?
# Try google
# Does that work for deep level objects like list of dictionaries?
#
alist = [1, 'person']
print(alist) # check the values for alist
# reset adictionary3
adict = { "name": "Einstein", "last": alist }
# len(adictionary3)
acopy1 = adict
acopy2 = adict.copy()

acopy2["name"] = Divya"
print(adict)
print(acopy2)


# %%
import copy
acopy3 = copy.copy(adict)

acopy1[2] = 'duo'

print(adict)
print("acopy1", acopy1)
print("acopy2",acopy2)
print("acopy3",acopy3)

 

# %%
# len(adictionary3)
acopy1 = adict
acopy2 = adict.copy()

import copy
acopy3 = copy.copy(adict)
acopy4 = copy.deepcopy(adict)

acopy1[2] = 'duo'
acopy1['last'][3]='nose'

print(adictionary3)
print("acopy1", acopy1)
print("acopy2",acopy2)
print("acopy3",acopy3)
print("acopy4",acopy4)