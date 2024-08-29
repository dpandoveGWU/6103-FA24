# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
# ## List comprehension  
# ### Basic
# \[output expression **for** iterator variable **in** iterable\]
# ### Advanced
# \[output expression + conditional **on** output **for** iterator variable **in** iterable + conditional **on** iterable\]
#

#%%
# list comprehension examples
#
alist = [ 1,4,0,2,3 ]
# First, using loops, to create a list from the list alist
listComp = []
for n in alist:
  listComp.append(2*n+1)

print(listComp)

#%%
# Next, use list comprehension in python
listComp = [ 2*n+1 for n in alist ]
print(listComp)
# result: [3, 9, 1, 5, 7]



#%%
# Nested loops
pairs = []
for n1 in range(4):
  for n2 in range(5,8):
    pairs.append( (n1,n2) ) # append them as tuples
print(pairs)

#%%
# or use list comprehension
pairs = [ (n1,n2) for n1 in range(4) for n2 in range(5,8) ]
print(pairs) 

# as with most one-liners, the convenience is sacrificed by readability of the codes. 
# more exposure and practice always help though.


#%%
grades = ['A','C+','B-','A-']

# first try old-fasion loops
gradeValues = []
for val in grades:
  if val=='A' : gradeValues.append(4)
  elif val=='A-' : gradeValues.append(3.7)
  elif val=='B+' : gradeValues.append(3.3)
  elif val=='B' : gradeValues.append(3)
  elif val=='B-' : gradeValues.append(2.7)
  elif val=='C+' : gradeValues.append(2.3)
  elif val=='C' : gradeValues.append(2)
  elif val=='C-' : gradeValues.append(1.7)
  elif val=='D' : gradeValues.append(1)
  else : gradeValues.append(0)

print(gradeValues)
# result: [4, 2.3, 2.7, 3.7]

#%%
# or 
# Conditionals in comprehensions
gradeValues = [ 4 if v=='A' else 3.7 if v=='A-' else 3.3 if v=='B+' else 3 if v=='B' else 2.7 if v=='B-' else 2.3 if v=='C+' else 2 if v=='C' else 1.7 if v=='C-' else 1 if v=='D' else 0 for v in grades ]
print(gradeValues)

#%%
# dictionary comprehensions
# Use curly brackets, and the key: value pair   { k:v for someVar in someIterable}
farmWords=['cow','pig','chicken']
lengthofwords = { key: len(key) for key in farmWords }

print(lengthofwords)
# result: {'cow': 3, 'pig': 3, 'chicken': 7}

#%%
# generator 
# review on list comprehension
alist = [ 1,4,0,2,3 ]
listComp = [ 2*n+1 for n in alist ]
print(listComp) # this is a list object
print(type(listComp))

#%%
# for generator, use ( ) instead of [ ] 
# ( ) here is not a tuple! It's a "(list) generator"
listGen = ( 2*n+1 for n in alist )
print(listGen) # this is a generator object
print(type(listGen))

#%%
# both list comprehension and generator objects can be iterated 
print(listGen)
for val in listGen: print(val) # essentially same as print(listComp), except it is not a list

#%%
# Generators vs List comprehensions

# #######  DO NOT do this
# repeat:  DO NOT DO THIS
#%%
# [n**2 for n in range(10**(100000)) ]
# Your wonderful computer cannot handle this. Be kind.
#

#%%
# But you can do this:
listGen = ( n**2 for n in range(10**1000000) )
print(listGen)
for v in listGen: 
  if v>99 : break
  print(v)

#%%
# OR 
listGen = ( n**2 for n in range(10**1000000) )
for i,v in enumerate(listGen):
  if v>99 : break
  print(i,": squared value is ",v)


#%%
# Generator Functions
# use yield instead of return
# they are functions that produces generators
# in other words, they produces iterable sequences

def num_seq(max):
  """Generate values from 0 to n"""
  i=0
  while i<max:
    yield i 
    i += 1

print(type(num_seq))
result = num_seq(6)
print(type(result))
for i in result: print(i)


#%% [markdown]
#
# If you can write useful recursive generator functions (for example, Fibonacci), you must be a true master of the art
#

