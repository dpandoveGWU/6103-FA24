#%%
import math 
import os 
print("Hello world!")

#%%
# Indentations more important than ever!
# loops - basic
for i in range(10):
  print(i)

#%%
# loops - basic
print("looping i:")
for i in range(1000):
  print('still going',i)
  if i>13:
    break # This break out from the loop, whatever loop the system is in.

#%%
print("\nlooping j:")  # Recall '\n' is the newline character, '\t' is a tab, etc.
for j in range(2,1000,2):
  if j<933:
    continue # This skips to the end of this iteration, and starts from the begining of the loop with the next iteration
  print(j)
  if j>945:
    break
  # if j<938: # Try setting it to 938 or 958, see the difference
  if j<958: # Try setting it to 938 or 958, see the difference
    continue
  print("Can you see me?")

#%%
# loops - iterate a list/tuple/set/dictionary, or anything "subscriptable"
# any difference among the three below?

# for val in list :
print("\nloop thru val in list:")
for val in [ 4,'2',("a",5),'end' ] :
  print(val, type(val))

# for val in tuple :
print("\nloop thru val in tuple:")
for val in ( 4,'2',("a",5),'end' ) :
  print(val, type(val))

# for val in set :
print("\nloop thru val in set:")
for val in { 4,'2',("a",5),'end' } :
  print(val, type(val))

#%%
# Now for dictionary
# for val in dictionary : (keys only)
print("\nloop thru key, val in dictionary??")
adictionary = { "k0":4, "k8":'2', "k1":("a",5), "k5":'end' }
for key in adictionary :
  print('key:', key, '; val', adictionary[key])

#%%
# or try this for dictionary, using .items() to get the pairs
print("\nalternative method to loop thru key, val in dictionary:")
# adictionary.items() # creates a object type of dict_items, which can be looped thru as key/value pairs   
for key, val in adictionary.items() :
  # print("key:", key, "value:", val, "type of value", type(val))
  print(f"key: {key}, value: {val}, and type of val: {type(val)}" )

#%%
# for val in string :
print("\nloop characters in a string:")
for char in 'GW Rocks' :
  print(char, type(char))
  
  
#%%
# Use enumerate function to generate the index??
# for index, val in enumerate(list) :
print("\nloop index value pairs in a list:")
alist = [ 4,'2',("a",5),'end' ]
for index, val in enumerate(alist) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print()

#%%
# Try tuple, set, and dictionary
print("\nloop key value pairs in a tuple like that?")
atuple = ( 4,'2',("a",5),'end' )
for index, val in enumerate(atuple) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print("OK\n")

print("\nloop key value pairs in a set like that?")
aset = { 4,'2',("a",5),'end' }
for index, val in enumerate(aset) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print("OK. BUT order is messed up!!\n")

#%%
print("\nloop key value pairs in a dictionary like that?")
adictionary = { "k0":4, "k8":'2', "k1":("a",5), "k5":'end' }
for index, val in enumerate(adictionary) :
  # print("index", index, "value", val, alist[index], type(val))
  print(f"index: {index}, value: {val}, and type of val: {type(val)}" )
print("Not quite what we want!!\n")


#%%[markdown]
# # Final thoughts on loops
# Writing loops for programmers is very basic. Should be able to do this in sleep.
# On the other hand, try to avoid explicitly writing out the loops if there are other 
# bulit-in alternatives. 
# 
# In particular, a lot of methods and functions in Numpy, Pandas, models, that are doing the loops 
# automatically. And most usually, those methods are faster (those are very efficient libraries 
# written in C++ and other efficient languages on the backend), easier to read, and easier to use. 
# Keep this in mind!!!
#
#%%