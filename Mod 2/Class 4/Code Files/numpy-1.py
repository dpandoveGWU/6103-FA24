# %%[markdown]
# # SciPy Family
#
# Python-based ecosystem [scipy.org](https://scipy.org)  
# 
# * SciPy Library - Fundamental library for scientific computing
# * NumPy - Numeric Python: Base N-dimensional array package
# * Pandas - Data structures (Dataframes) & analysis
# * Matplotlib - Comprehensive 2D Plotting
# * Sympy - Symbolic mathematics
# * IPython - Enhanced Interactive Console
#
# Documentation: https://numpy.org/doc/1.26/
#
# The datatypes (dtype attribute) supported by Numpy is many:  
# [Numpy basic data types](https://docs.scipy.org/doc/numpy/user/basics.types.html) 


# %%
# might need to install numpy from the terminal
# %pip install numpy
# %pip3 install numpy
# %conda install numpy
# %pip freeze
# %pip list
# %pip show numpy


#%%
import numpy as np 
# or from numpy import *  
# import matplotlib.pyplot as plt
# import pandas as pd  
np.__version__


# %%
# Discussion Activity:
#
# Why use matrices for data science?
# - 
# -
# -



# %%
#
# Lists are 1D matrices
list0a = [9,8,7]
list0b = [6,5,4]
# What do you get when you sum two lists?
list0a + list0b


# %% 
# Also try this
2 * [0, 1, 2, 3]


# %%
# Utilizing this behavior, we can initialize constant lists of a given size
[None] * 10



# %%
# explore data structures with list of list, how many dimensions? 
list1a = [ [11,12,13,14], [21,22,23,24], [31,32,33,34]] 
list1b = [ [41,42,43,44], [51,52,53,54], [61,62,63,64]] 
#


#%%
# Again, what is list1a + list1b?
list2D = list1a + list1b
print( list2D )


#%%
# Activity: 1
# Question: How do you describe (in english) these two lists? What are the 
# "shapes" of the objects?

# - These are 3 by 4 matrices. Two-dimensional arrays. 
# Question: how do you get the element '32' in list5?
list2D[2][1]


# %%
# Activity: 2
# Question: how do you get the row of [31,32,33,34] in list5?
list2D[2]



#%%
# Activity: 3
# Question: How to you get the first column ???
# List comprehension: https://realpython.com/list-comprehension-python/
[ row[1] for row in list2D ]



#%%
# OR Loop it
v3 = []
for row in list2D: 
  v3.append(row[1])
print(v3)


#%%
# How do you create a higher-dimensional list (say 2x3x4)?
# 
# list3D = [ [ [111,112,113], [121,122,123], [131,132,133], [141,142,143] ] 
# , [ [211,212,213], [221,222,223], [231,232,233], [241,242,243] ] ] 

list3D = [ [ [ 111, 112, 113, 114 ], [ 121, 122, 123, 124 ], [131, 132, 133, 134] ] , 
           [ [ 211, 212, 213, 214 ], [ 221, 222, 223, 224 ], [231, 232, 233, 234] ] ]

print( list3D )



#%%
# Now try numpy
import numpy as np


# %%
# Numpy provides a data structure (class) known as nd-array or simply numpy
# array
# Some basic attributes and simple methods of numpy arrays


# %%
# Use np to create a range
arr = np.arange(15) 
print(f'a:\n{arr}\n')


# %%
# Get the type of `arr`
type(arr)


# %%
# Question: what is `arange`?
# A function or a method? 


# %%
# Explore the attributes and methods of `ndarray`
arr.shape # `shape` is a class attribute


# %%
# `arr` is a 1-dimensional array or vector
# Is it a row vector or a column vector?



# %%
# Reshape a matrix
arr1 = np.arange(20)
arr2 = arr1.reshape(4, 5) # Using -1 for the last dimension lets numpy 

print(arr1) # the `reshape` method does not alter the object
print(arr2)


# %%
# Calculate the new shape of the array
arr2.shape


# %%
# Let's turn our 3D list into a 3D numpy array
arr3D = np.array(list3D)
print( arr3D )


# %%
# Guess the output from the shape attribute
arr3D.shape


# %%
# Read the API here: https://numpy.org/doc/stable/reference/index.html
a = np.arange(20).reshape(2,5,-1)
print(f'a.shape: {a.shape}') 
print(f'a.ndim: {a.ndim}') 
print(f'a.dtype: {a.dtype}') 
print(f'a.dtype.name: {a.dtype.name}') 
print(f'a.itemsize: {a.itemsize}') 
print(f'a.size: {a.size}') 
print(f'type(a): {type(a)}') 
# print( a )


# %%
# Try to same with this array
b = np.array([6, 7.0, 8])
print(f'a.shape: {b.shape}') 
print(f'a.ndim: {b.ndim}') 
print(f'a.dtype: {b.dtype}') 
print(f'a.dtype.name: {b.dtype.name}') 
print(f'a.itemsize: {b.itemsize}') 
print(f'a.size: {b.size}') 
print(f'type(a): {type(b)}') 
# print( b )
#


# %%
# The opposite of reshape, can use ravel()
print(f'a ravel: {a.ravel().shape}')
print(f'a again: {a}') 



# %%
# IMPORTANT
# The a.ravel() function does NOT change a!! 
# I create a true copy of a and ravel/unravel it only. 
# Remember the differences in class/object definitions, 
# it is critical what is the "return" value in 
# those function/methods. 
# If return self, you are getting back the object a. 
# But this function return a separate true copy of 
# the result instead. This is by design. 
# 
# A lot of other functions in numpy/pandas behave like that too. 
# They do NOT change the original object (which is more practical 
# in most cases). It might return a copy of the object after operation, 
# or just return None.
# 



#%%
# If you really want to change a, try this: 
# a = a.ravel() # exact same result as 
# In other words, reassign the result as the original named object. 
a = a.reshape(-1)
print(f'a: {a}')
print(f'type a: {type(a)}')
print(f'a.shape: {a.shape}') 



#%%
# Other examples to create some simply numpy arrays
# Question: where to find them in the API?
#
print(f'zeros: {np.zeros( (3,4) )}')
print(f'ones: {np.ones( (2,3,4), dtype = np.int16 )}')
print(f'empty: {np.empty( (2,3) )}')
print(f'arange variation 1: {np.arange( 10, 30, 5 )}')
print(f'arange variation 2: {np.arange( 0, 2, 0.3 )}' )
print(f'complex: {np.array( [ [1, 2], [3, 4] ], dtype=complex )}')
print(f'float: {np.arange(2, 10, dtype=float)}')



# %%
# Linear space
from numpy import pi
import matplotlib.pyplot as plt # for plotting

n = 50
x = np.linspace( 0, 2*pi, n )
f = np.sin(x) # np sine function will broadcast
plt.plot(x, f)
plt.show()




# %%
# Type inference
list = [ 5, 'a', 2, 3.5, True ]

nparray = np.array(list)
print(f"nparray = \n{nparray}")
print(f'nparray: shape - {nparray.shape} , type - {type(nparray)}')
print(f"nparray.dtype = {nparray.dtype}") 



# %%
list = [ 5, [1,4], 3, 1 ]
nparray = np.array(list)
print(f"nparray = \n{nparray}")
print(f'nparray: shape - {nparray.shape} , type - {type(nparray)}')
print(f"nparray.dtype = {nparray.dtype}") 



# %%
# Matrix addition
nparray1 = np.array([0, -2, 3.0])
nparray2 = np.array([0, 3.0, '1'])

try:
  nparray12 = nparray1 + nparray2
  print( nparray12 )
except ValueError as err :
  print("Value Error: {0}".format(err))
except TypeError as err :
  print("Type Error: {0}".format(err))



# %%
# Matrix dot product
# If they are 2D-arrays, and have compatible dimensions, you can multiply them 
# as matrices
nparray1 = np.array([0, -2, 3.0])
nparray2 = np.array([0, 3.0, 1])

prod = np.dot(nparray1, nparray2)
print(f"prod = {prod}")
print(f"prod.shape = {prod.shape}")


# %%
# Try the dot product of a 2D array and a vector


# %%
# Also try the 3d-array that we constructed...
# In physics, those are called tensors. 
nparray3D = np.array(list3D)
print(f"nparray3D = \n{nparray3D}")
print(f'nparray3D: shape - {nparray3D.shape} , type - {type(nparray3D)}')
print(f"nparray3D.dtype = {nparray3D.dtype}") 


#%%
# If they have compatible dimensions, you can multiply them as matrices
nparray = np.array([0, 1, 2, 3])
prod1 = np.dot(nparray3D, nparray)
print(f"prod1.shape = {prod1.shape}")


#%%[markdown]
# speed and ease of use is the strength of numpy arrays, compared to python lists. 
# The entire array must be of a single type, however.
# If we try to time or clock the code execution times, you will find similar functions 
# is much faster than looping thru a python list.
# This is mainly because NumPy is written in C, and optimized these specialized 
# operations in a well-designed library.
# That's why I said, although looping is fundamental in programming, avoid using 
# them explicitly in python coding unless there are no better options. 


# %%
# filtering and indexing
arr = np.arange(20).reshape(4, 5)
print( arr )

# %% Access rows or columns
# Take a guess
arr[1]
# arr[1][2]
# arr[:, 1:2]


# %%
print(arr[:, 2])
#print(arr[:, -1:])


#%%
# Let us do something simpler.
# Obtain the third column of nparray1
print(nparray1)
v3 = nparray1[:,2]
print(v3) 
print(v3.shape) # it is a column vector, or array one by three (3,1)

# Much easier than dealing with lists on the coding side of things. Speed is 
# also maximized.


#%%
# ################################################
#          BROADCASTING   (!!!Important!!!)
# ################################################
# 
# Let's practice slicing numpy arrays and using NumPy's broadcasting concept. 
# Remember, broadcasting refers to a numpy array's ability to VECTORIZE 
# operations, so they are performed on all elements of an object at once.
# If you need to perform some simple operations on all array elements, 
# %%
nparray1squared = arr ** 2
print(nparray1squared)

#%%
nparray1mod7 = arr % 7 # remainder from dividing by 7
print(nparray1mod7)


#%%
nparray1b = np.array(list1b)
nparray1bovera = nparray1b / nparray1
print(nparray1bovera)

# Try some other operations, see if they work.

# Next try to do the above with loops or comprehensions? 

#%%
# boolean indexing 
print(nparray1)
npbool1greater = nparray1 > 21
print(npbool1greater)


#%%
print(nparray1[npbool1greater])


#%%
print(nparray1[npbool1greater].shape)


#%%
npbool1mod = nparray1 % 2 == 1
print(npbool1mod)
print(nparray1[npbool1mod])
print(nparray1[npbool1mod].shape)


#%%
# Some other basic numpy operations
a = np.array( [20,30,40,50] )
b = np.arange( 4 )
print(f'b: {b}')
c = a-b
print(f'c: {c}')
print(f'b**2: {b**2}')
print(f'sine: {10*np.sin(a)}')
print(f'#{50*"-"}')


#%%
a = np.random.random((2,3))
print(f'a:\n{a}\n')
print(f'a.sum: {a.sum()}')
print(f'a.min: {a.min()}')
print(f'a.max: {a.max()}')
print(f'#{50*"-"}')


# %%
b = np.arange(24).reshape(3,4,2)
print(f'b: {b}')
print(f'b.sum axis 0: {b.sum(axis = 0)}') # sum of each column
print(f'b.min axis 1: {b.sum(axis = 1)}') # min of each row
print(f'b.max axis 2: {b.sum(axis = 2)}') # min of each row
print(f'b.cumsum axis 1: {b.cumsum(axis = 1)}') # cumulative sum along each row
print(f'#{50*"-"}')


#%% 
# We looked at mostly the functions/methods within the numpy object, like 
# a.sum()
# a.min() etc.
# As we discussed in OOP class, a lot of times, we can also have methods for the 
# class itself. 
# In this case, we can use the universal functions in the numpy library directly, 
# like np.exp(), np.sqrt(). np.add() etc.
a = np.arange(3)
print(f'a:\n{a}\n')
e = np.exp(a); print(f'exp: {e}')
rt = np.sqrt(a); print(f'root: {rt}')
b = np.array([2., -1., 4.])
sum = np.add(a, b)
print(f'sum: {sum}')
print(f'#{50*"-"}')

#%% 
# Indexing and slicing in Numpy
a = np.arange(10)**3; 
print(f'a: {a}')
print(f'a[2]: {a[2]}')
print(f'a[2:5]: {a[2:5]}') # same as basic python syntax, not including 5.
a[:6:2] = -1000
print(f'a[::-1] : {a[ : :-1]}')
# reset a
a = np.arange(10)**3
for i in a:
  print(f'i-cube rt: {np.power(i,1/3)}')
#
