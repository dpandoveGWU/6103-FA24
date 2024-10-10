# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%% [markdown]
#
# HW Numpy 
# ## By: Wali Siddiqui
# ### Date: 4th Oct 2024
#


#%%
# NumPy

import numpy as np

# %%

# %%
# ######  QUESTION 1      QUESTION 1      QUESTION 1   ##########
# This exercise is to test true/shallow copies, and related concepts. 
# ----------------------------------------------------------------
# 
# ######  Part 1a      Part 1a      Part 1a   ##########
# 
list2 = [ [11,12,13], [21,22,23], [31,32,33], [41,42,43] ] # two dimensional list (2-D array)  # (4,3)
nparray2 = np.array(list2)
print("nparray2:", nparray2)

# We will explain more of this indices function in class next. See notes in Class05_02_NumpyCont.py
idlabels = np.indices( (4,3) ) 
print("idlabels:", idlabels)

i,j = idlabels  # idlabels is a tuple of length 2. We'll call those i and j
nparray2b = 10*i+j+11
print("nparray2b:",nparray2b)

# 1.a) Is nparray2 and nparray2b the "same"? Use the logical "==" test and the "is" test. 
# Write your codes, 
# and describe what you find.

print(nparray2 == nparray2)

# (Answer) based on the above logical operation the two arrays are the same as all elements retun to true indicating all elemets within yhe array are equal 

# %%
# ######  Part 1b      Part 1b      Part 1b   ##########
# 
# 1.b) What kind of object is i, j, and idlabels? Their shapes? Data types? Strides?
# 
# write your codes here
print("idlabels is an object of type:", type(idlabels))
print("i is an object of type:", type(i))
print("j is an object of type:", type(j))
print("idlabels has a shape of:", idlabels.shape)
print("i has a shape of:", i.shape)
print("j has a shape of:", j.shape)
print("idlabels is of data type:", idlabels.dtype)
print("i is of data type:", i.dtype)
print("j is of data type:", j.dtype)
print("idlabels has the following strides:", idlabels.strides)
print("i has the following strides:", i.strides)
print("j has the following strides:", j.strides)

# %%
# ######  Part 1c      Part 1c      Part 1c   ##########
# 
# 1.c) If you change the value of i[0,0] to, say 8, print out the values for i and idlabels, both 
# before and after the change.
# 
# write your codes here
print("Value of i before the change:", i)
print("Value of idlabels before the change:", idlabels)

i[0, 0] = 8

print("Value of i after the change:", i)
print("Value of idlables after the change:", idlabels)

# Describe what you find. Is that what you expect?
# There is a change to the first element in i (0, 0) which is not replaced to 8 instead of the orginal
# zero. If we also check idlabels the first matrix that is the row index matrix has also changed similar to thta of i
# Also try to change i[0] = 8. Print out the i and idlabels again.
i[0] = 8
print("i:", i)
print("idlabels:", idlabels)

# %%
# ######  Part 1d      Part 1d      Part 1d   ##########
# 
# 1.d) Let us focus on nparray2 now. (It has the same values as nparray2b.) 
# Make a shallow copy nparray2 as nparray2c
# now change nparray2c 1,1 position to 0. Check nparray2 and nparray2c again. 
# Print out the two arrays now. Is that what you expect?
# 
# Also use the "==" operator and "is" operator to test the 2 arrays. 
# write your codes here
# shallow copy
nparray2c = nparray2.copy()
nparray2c[1, 1] = 0

# check matrices
print("nparray2:", nparray2)
print("nparray2c:", nparray2c)

# since we make a shallow copy of nparray2 and store it in nparray2c, changing an element of nparray2c
# does not have any effect on the elemets of nparray2

print("nparray2 == nparray2c:", nparray2 == nparray2c) 
print("nparray2 is nparray2c:", nparray2 is nparray2c) 

# The elementwise comparison will show that the arrays are not fully equal because of the change at position [1,1]
# The is operator between the two arrays will retun Flase since both the arrays are different objects
#%%
# ######  Part 1e      Part 1e      Part 1e   ##########
# Let us try again this time using the intrinsic .copy() function of numpy array objects. 
nparray2 = np.array(list2) # reset the values. list2 was never changed.
nparray2c = nparray2.copy() 
# now change nparray2c 0,2 position value to -1. Check nparray2 and nparray2c again.
# Are they true copies?
# 
# write your codes here
# Again use the "==" operator and "is" operator to test the 2 arrays. 
#
# Since numpy can only have an array with all values of the same type, we usually 
# do not need to worry about deep levels copying. 
# 
# ######  END of QUESTION 1    ###   END of QUESTION 1   ##########




# %%
# ######  QUESTION 2      QUESTION 2      QUESTION 2   ##########
# Write NumPy code to test if two arrays are element-wise equal
# within a (standard) tolerance.
# between the pairs of arrays/lists: [1e10,1e-7] and [1.00001e10,1e-8]
# between the pairs of arrays/lists: [1e10,1e-8] and [1.00001e10,1e-9]
# between the pairs of arrays/lists: [1e10,1e-8] and [1.0001e10,1e-9]
# Try just google what function to use to test numpy arrays within a tolerance.



# ######  END of QUESTION 2    ###   END of QUESTION 2   ##########


# %%
# ######  QUESTION 3      QUESTION 3      QUESTION 3   ##########
# Write NumPy code to reverse (flip) an array (first element becomes last).
x = np.arange(12, 38)


# ######  END of QUESTION 3    ###   END of QUESTION 3   ##########


# %%
# ######  QUESTION 4      QUESTION 4      QUESTION 4   ##########
# First write NumPy code to create an 7x7 array with ones.
# Then change all the "inside" ones to zeros. (Leave the first 
# and last rows untouched, for all other rows, the first and last 
# values untouched.) 
# This way, when the array is finalized and printe out, it looks like 
# a square boundary with ones, and all zeros inside. 
# ----------------------------------------------------------------


# ######  END of QUESTION 4    ###   END of QUESTION 4   ##########



# %%
# ######  QUESTION 5      QUESTION 5      QUESTION 5   ##########
# Broadcasting, Boolean arrays and Boolean indexing.
# ----------------------------------------------------------------
i=3642
myarray = np.arange(i,i+6*11).reshape(6,11)
print(myarray)
# 
# a) Obtain a boolean matrix of the same dimension, indicating if 
# the value is divisible by 7. 


# b) Next get the list/array of those values of multiples of 7 in that original array  

# ######  END of QUESTION 5    ###   END of QUESTION 5   ##########





#
# The following exercises are  
# from https://www.machinelearningplus.com/python/101-numpy-exercises-python/ 
# and https://www.w3resource.com/python-exercises/numpy/index-array.php
# Complete the following tasks
# 

# ######  QUESTION 6      QUESTION 6      QUESTION 6   ##########

#%%
flatlist = list(range(1,25))
print(flatlist) 

#%%
# 6.1) create a numpy array from flatlist, call it nparray1. What is the shape of nparray1?
# remember to print the result
#
# write your codes here
#

#%%
# 6.2) reshape nparray1 into a 3x8 numpy array, call it nparray2
# remember to print the result
#
# write your codes here
#

#%%
# 6.3) swap columns 0 and 2 of nparray2, and call it nparray3
# remember to print the result
#
# write your codes here
#

#%%
# 6.4) swap rows 0 and 1 of nparray3, and call it nparray4
# remember to print the result
#
# write your codes here
#

#%%
# 6.5) reshape nparray4 into a 2x3x4 numpy array, call it nparray3D
# remember to print the result
#
# write your codes here
#

#%%
# 6.6) from nparray3D, create a numpy array with boolean values True/False, whether 
# the value is a multiple of three. Call this nparray5
# remember to print the result
# 
# write your codes here
#

#%%
# 6.7) from nparray5 and nparray3D, filter out the elements that are divisible 
# by 3, and save it as nparray6a. What is the shape of nparray6a?
# remember to print the result
#
# write your codes here
#

#%%
# 6.8) Instead of getting a flat array structure, can you try to perform the filtering 
# in 6.7, but resulting in a numpy array the same shape as nparray3D? Say if a number 
# is divisible by 3, keep it. If not, replace by zero. Try.
# Save the result as nparray6b
# remember to print the result
# 
# write your codes here
#
# 
# ######  END of QUESTION 6    ###   END of QUESTION 6   ##########

#%%
#
