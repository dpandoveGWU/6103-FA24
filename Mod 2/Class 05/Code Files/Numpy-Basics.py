#Numpy Basics
# %%
# might need to install numpy from the terminal
# %pip install numpy
# %pip3 install numpy
# %conda install numpy
# %pip freeze
# %pip list
# %pip show numpy

# %% 
import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

#%%
#
import numpy as np

my_arr = np.arange(1000000)
my_list = list(range(1000000))

#%%
%timeit my_arr2 = my_arr * 2
%timeit my_list2 = [x * 2 for x in my_list]

#%%
# Create Arrays
import numpy as np
data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
data
#%%
#Shape and Type 
print(data.shape)
print(data.dtype)
#%%
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)
#%%
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
print(arr2)
#%%
print(arr2.ndim)
print(arr2.shape)
#%%
print(arr1.dtype)
print(arr2.dtype)
#%%
np.zeros(10)
np.zeros((3, 6))
np.empty((2, 3, 2))
#%%
np.arange(15)
#%%
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
print(arr1.dtype)
print(arr2.dtype)

#%%
arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
float_arr = arr.astype(np.float64)
print(float_arr)
print(float_arr.dtype)
#%%
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
print(arr)
arr.astype(np.int32)
#%%
numeric_strings = np.array(["1.25", "-9.6", "42"], dtype=np.string_)
numeric_strings.astype(float)

#%%
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)
#%%
zeros_uint32 = np.zeros(8, dtype="u4")
print(zeros_uint32)
#%%
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr)
print(arr * arr)
print(arr - arr)
#%%
print(1 / arr)
print(arr ** 2)
#%%
arr2 = np.array([[0., 4., 1.], [7., 2., 12.]])
print(arr2)
print(arr2 > arr)
#%%
#Slicing
arr = np.arange(10)
print(arr)
print(arr[5])
print(arr[5:8])
arr[5:8] = 12
print(arr)
#%%
arr_slice = arr[5:8]
print(arr_slice)
#%%
arr_slice[1] = 12345
print(arr)
#%%
arr_slice[:] = 64
print(arr)
#%%
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2])
#%%
print(arr2d[0][2])
print(arr2d[0, 2])
#%%
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d)
#%%
print(arr3d[0])
#%%
old_values = arr3d[0].copy()
arr3d[0] = 42
print(arr3d)
arr3d[0] = old_values
print(arr3d)
#%%
arr3d[1, 0]
#%%
x = arr3d[1]
print(x)
print(x[0])
#%%
print(arr)
arr[1:6]

#%%
print(arr2d)
print(arr2d[:2])

#%%
print(arr2d[:2, 1:])

#%%

lower_dim_slice = arr2d[1, :2]
#%%

print(lower_dim_slice.shape)
#%%

print(arr2d[:2, 2])
#%%
print(arr2d[:, :1])

#%%
arr2d[:2, 1:] = 0
print(arr2d)

#%%
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2],
                 [-12, -4], [3, 4]])
print(names)
print(data)

#%%
#Transposing Arrays and Swapping Axes

arr = np.arange(15).reshape((3, 5))
print(arr)
print(arr.T)

#%%
#Dot matrix computations
arr = np.array([[0, 1, 0], [1, 2, -2], [6, 3, 2], [-1, 0, -1], [1, 0, 1]])
print(arr)
print(arr.T)
print(np.dot(arr.T, arr))
# (0×0)+(1×1)+(6×6)+(−1×−1)+(1×1)=0+1+36+1+1=39

#%%
#Swap the indicated axes to rearrange the data
print(arr)
print(arr.swapaxes(0, 1))

#%%
# Pseudorandom Number Generation
samples = np.random.standard_normal(size=(4, 4))
print(samples)

#%%
from random import normalvariate
N = 1000000
%timeit samples = [normalvariate(0, 1) for _ in range(N)]
%timeit np.random.standard_normal(N)

#%%
rng = np.random.default_rng(seed=12345)
data = rng.standard_normal((2, 3))
#%%
type(rng)

#%%
#Unarry ufuncs
arr = np.arange(10)
print(arr)
print(np.sqrt(arr))
print(np.exp(arr))

#%%
#Binary ufuncs
x = rng.standard_normal(8)
y = rng.standard_normal(8)
print(x)
print(y)
print(np.maximum(x, y))

#%%
#Array oriented progamming with Arrays
#Example: evaluate the function sqrt(x^2 + y^2) across a regular grid of values.
points = np.arange(-5, 5, 0.01) # 100 equally spaced points
print(points)
xs, ys = np.meshgrid(points, points)
print(ys)
print(xs)
#%%
#evaluating the function is a matter of writing the same expression you would write with two point
z = np.sqrt(xs ** 2 + ys ** 2)
print(z)


# %%
#Use matplotlib to create visualizations of this two-dimensional array:
import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray, extent=[-5, 5, -5, 5])
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")

# %%
#Conditional logic as Array operations
#Suppose we had a Boolean array and two arrays of values:
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# %%
#Suppose we wanted to take a value from xarr whenever the corresponding value in cond is True, 
# and otherwise take the value from yarr

result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
print(result)

# %%
result = np.where(cond, xarr, yarr)
print(result)
# %%
#Sorting
arr = rng.standard_normal(6)
print(arr)
arr.sort()
print(arr)

# %%
#sorting array along an axis by passing axis number
arr = rng.standard_normal((5, 3))
print(arr)
arr.sort(axis=0)
print(arr)
arr.sort(axis=1)
print(arr)
# %%
#Linea Algebra
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print("x:",x)
print("y:",y)
print("dot product:",x.dot(y))

# %%
#x.dot(y) is equivalent to np.dot(x, y):
np.dot(x, y)
# %%
#A matrix product between a two-dimensional array and a suitably sized one-dimensional array results in a one-dimensional array:
x @ np.ones(3)

#%%
#numpy.linalg has a standard set of matrix decompositions and things 
# like inverse and determinant:

from numpy.linalg import inv, qr
X = rng.standard_normal((5, 5))
mat = X.T @ X
inv(mat)
mat @ inv(mat)

#%%

