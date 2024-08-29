#%%
import math 
import os 
print("Hello world!")


#%%[markdown]
# # Logic
# ## Conditional statment
# 
# _________________________________________________  
# Statement:     If p, then q     OR   p  &rarr;  q   
#
# Contrapositve: If -q, then -p   OR   -q &rarr; -p  
# _________________________________________________  
# Inverse:       If -p, then -q   OR   -p &rarr;  -q   
# 
# Converse:      If q, then p     OR   q  &rarr;  p  
# _________________________________________________  

#%%[markdown]
# ## Some other logic rules
# 
# _________________________________________________  
# -(p AND q) \
# &harr; \
# -p OR -q 
# _________________________________________________  
# -(p OR q) \ 
# &harr; \
# -p AND -q 
# _________________________________________________  
# p AND (q AND r) \
# &harr; \
# (p AND q) AND r 
#
# we usually combine as \
# (p AND q AND r)
# _________________________________________________  
# p OR (q OR r) \
# &harr; \
# (p OR q) OR r 
#
# we usually combine as \
# (p OR q OR r)
# _________________________________________________  
# ## Distributive law 1
# p AND (q OR r) \
# &harr; \
# (p AND q) OR (p AND r) 
# _________________________________________________  
# ## Distributive law 2
# p OR (q AND r) \
# &harr; \
# (p OR q) AND (p OR r) 
# _________________________________________________  
#

#%%
# #######################################################
# PAY ATTENTION to Indentation NOW !!! 
# #######################################################
# Basic logic
x = 1
y = 2
b = (x == 1)
b = (x != 1)
b = (x == 1 and y == 2)
b = (x != 1 and y == 2)
b = (x == 1 and y != 2)
b = (x != 1 and y != 2)
b = (x == 1 or y == 2)
b = (x != 1 or y == 2)
b = (x == 1 or y != 2)
b = (x != 1 or y != 2)
if y == 1 or 2 or 3:
	print("OK")
print( y == 1 or 2 or 3 )
if y == 1 or 3 or 5:
  print("OK")
else:
  print('NOT okay.')

# The above is NOT producing what we want!!!
# Fix it!


#%%
# conditional
# if :
income = 60000
if income >100000 :
  print("rich")
  
# if else:
if income >100000 :
  print("rich")
else :
  print("not rich")
  
#%%
# if elif elif .... :
if income >200000 :
  print("super rich")
elif income > 100000 :
  print("rich")
elif income > 40000 :
  print("not bad")
elif income > 0 :
  print("could be better")
else :
  print("no idea")

#%%
# The above can be compacted into a one-liner
print("super rich" if income > 200000 else "rich" if income > 100000 else "not bad" if income > 40000 else "could be better" if income > 0 else "no idea" )
# or 
incomelevel = "super rich" if income > 200000 else "rich" if income > 100000 else "not bad" if income > 40000 else "could be better" if income > 0 else "no idea" 
print(incomelevel)

# write your conditional statment to assign letter grades A, A-, B+ etc according to the syllabus


#%%
# loops - basic
for i in range(10):
  print(i)

#%%
# loops - basic
print("looping i:")
for i in range(1000):
  print('still going',i)
  if i>13:
    break

#%%
print("\nlooping j:")  # '\n' is printing a newline character; \t is a tab, etc. 
for j in range(2,1000,2):
  if j<933:
    continue
  print(j)
  if j>945:
    break
  # if j<938: # Try setting it to 938 or 958, see the difference
  if j<958: # Try setting it to 938 or 958, see the difference
    continue
  print("Can you see me?")

#%%
# loops - iterate a list/tuple/set/dictionary
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


# %%
