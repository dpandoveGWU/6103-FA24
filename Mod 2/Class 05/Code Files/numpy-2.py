# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
# Numpy continued
import numpy as np 

#%%
# Stacking numpy arrays
a = np.floor(10*np.random.random((2,3)));print(f'a:\n{a}\n')
b = np.floor(10*np.random.random((2,3)));print(f'b:\n{b}\n')
# 
z = np.vstack((a,b)); print(f'vstack z:\n{z}\n')
z1 = np.hstack((a,b)); print(f'hstack z1:\n{z1}\n')
z2 = np.column_stack((a,b)) 
print(f'column_stack z2:\n{z2}')
# column_stack same as hstack between two numpy arrays here
# the two has different behavior acting on lists
# try comment in the python list defs for a and b above. 
print(f'#{50*"-"}')

#%%
# newaxis
from numpy import newaxis
a = np.array([4.,2.,6.])
b = np.array([3.,7.,5.])
print(f'a[:,newaxis] :\n{a[:,newaxis]}\n')
z3= np.column_stack((a[:,newaxis],b[:,newaxis]))
print(f'z3:\n{z3}\n')
z3b= np.hstack((a[:,newaxis],b[:,newaxis]))
print(f'z3b:\n{z3b}\n')
z4= np.vstack((a[:,newaxis],b[:,newaxis])); 
print(f'z4:\n{z4}')
print(f'#{50*"-"}')

#%%
# Splitting Numpy arrays
a = np.floor(10*np.random.random((2,12)))
print(f'a:\n{a}\n')
z1 = np.hsplit(a,3) # Split a into 3. If dimension not divisible -> ValueError
print(f'z1: len - {len(z1)} , type - {type(z1)}\n')
for elt in z1:
  print(f'elt:\n',elt)

#%%
# Split a after the third and the fourth column
print(f'a:',a)
z2 = np.hsplit(a,(3,4))
# print(f'type(z1): {type(z1)}')
print(f'len(z2):',len(z2))
for elt in z2:
  print(f'elt:\n',elt)
print(f'#{50*"-"}')

#%%
# The .ix_() function ( I interpret it as Index-eXtraction )
# https://numpy.org/doc/stable/reference/generated/numpy.ix_.html 
#
e = np.arange(25).reshape(5,5)
print(f'e:\n{e}\n')
# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]
#  [15 16 17 18 19]
#  [20 21 22 23 24]]

sub_indices = np.ix_([1,3],[0,4]) # If your ultimate ndarray is rank 3 (m x n x k), then you will need three arguments
print(f'type(sub_indices) : {type(sub_indices)}')
print(f'len(sub_indices) : {len(sub_indices)}') # In other words, for rank 3 ndarray, you'll need tuple here of length 3.
print(f'sub_indices :\n{sub_indices}\n')

print(f'type(sub_indices[0]) : {type(sub_indices[0])}')
print(f'sub_indices[0].shape : {sub_indices[0].shape}')
print(f'type(sub_indices[1]) : {type(sub_indices[1])}')
print(f'sub_indices[1].shape : {sub_indices[1].shape}')

print(f'\ne[sub_indices]:\n', e[sub_indices])

#%%
# rank 3 case
a = np.array([4,2,3])
b = np.array([5,4])
c = np.array([5,4,0,3])
ax,bx,cx = np.ix_(a,b,c) # This separates out the [0], [1], [2] elements of the tuple in one step

print(f'ax: {ax}'); print(f'bx: {bx}'); print(f'cx: {cx}')
print(f'shapes: {ax.shape}, {bx.shape}, {cx.shape}\n')
result = ax+bx*cx
print(f'1. result ax+bx*cx: {result}')
print(f'2. result[2,1,3]: {result[2,1,3]}')
print(f'3. individual a[2]+b[1]*c[3]: {a[2]+b[1]*c[3]}\n') # same as above

# More commonly, we use it for filtering like our previous rank 2 example.
e = np.arange(6*6*6).reshape(6,6,6)  #216 elements
print(f'This is e[(ax,bx,cx)]:\n{e[(ax,bx,cx)]}')



#%%
# Automatic Reshaping
a = np.arange(30)
print('Original- a.shape:',a.shape); print('Original- a:',a)
a.shape = 2,-1,3
print('After- a.shape:',a.shape); print('After- a:',a)
# Unlike the reshape() method, this way changes a directly!!
print(f'#{50*"-"}')
#%%
# Other functions worth learning:
# https://numpy.org/doc/stable/reference/generated/numpy.indices.html
idlabels =  np.indices( (5,4) ) 
print(f'idlabels: shape - {idlabels.shape} , type - {type(idlabels)}\n')

# This creates a set of i, j values 
# indicating the row/column indices, with dimensions 3 by 4
i, j = idlabels
print(f'i: shape - {i.shape} , type - {type(i)}')
print(f'This is i: \n{i}\n')
print(f'j: shape - {j.shape} , type - {type(j)}')
print(f'This is j: \n{j}\n')
# We can try, for example, create a matrix of M, whose values are 3*i - 2*j
m = 2*i - 2*j
print(f'This is 2*i - 2*j (broadcasting) {m}')

#%%
#

#%%
# Examples from https://www.machinelearningplus.com/python/101-numpy-exercises-python/ 
# and https://www.w3resource.com/python-exercises/numpy/index-array.php

#%%