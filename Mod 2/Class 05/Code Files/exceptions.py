# Exception Handling
# Run-time errors are called exceptions in Python


# %%
# Try to run the following line
1 / 0
# The code throws `ZeroDivisionError` exception
# Since we did not catch it, the execution stopped!


# %%
import math
def quadratic(a, b, c):
  discRoot = math.sqrt (b * b - 4 * a * c)

  root1 = (-b + discRoot) / (2 * a)
  root2 = (-b - discRoot) / (2 * a)

  return (root1, root2)


# %%
# Question: what are some possible ways the function execution go wrong? 
#
#
#


# %%
# If the exceptions are not handled, the entire program stops.
# Try



# %%
# Handling Exceptions
# Handle only non-fatal exceptions
# . . . . 
a, b, c = 1, 0, 1 # (possibly noisy) signal from the sensors

try:
  roots = quadratic(a, b, c) # This raised exceptions

except Exception: # Base non-fatal exception
  print('These are default roots!') # Log the error or not
  roots = (0, 0) # Some default value

finally: # Will execute no matter what
  print('We are about to land on the Moon!!')

print('We just landed on the Moon!!!') # The mission does not stop




# %%
# You can specify different exceptions, depending on your use case
a, b, c = 1, 0, 1 # (possibly noisy) signal from the sensors

try:
  roots = quadratic(a, b, c) # This raised exceptions
  print(roots)
  
except ValueError: # Base non-fatal exception
  print('These are default roots!') # Log the error, or not
  roots = (None, None) # Assign some default values, or not

except TypeError:
  print('Check the sensors!') # Log the error or not

finally: # Will execute no matter what
  print('We are about to land on the Moon!!')

print('We just landed on the Moon!!!') # The mission does not stop



# %%
# Reading a file: https://docs.python.org/3/library/functions.html#open
import sys
try:
    f = open('myfile.txt') # Open the file
    s = f.readline() # Read the first line
    i = int(s.strip()) # Covert the line to an int
except OSError as err:
    # Read another file or a default file, etc...
    i = -1
    print("OS error:", err)
except ValueError:
    # Assign a default value, etc...
    i = -1
    print("Could not convert data to an integer.")
except Exception as err:
    print(f"Unexpected {err=}, {type(err)=}")
    raise

print(i)



# %%
# Throwing exceptions
# Read: https://docs.python.org/3/library/exceptions.html
def factorial(n):
  if type(n) != int or n < 0:
    raise ValueError # raise an exception
  
  prod = 1
  for i in range(1, n + 1):
    prod *= i
  return prod



# %%
# We need to signal the user program that something went wrong
# . . .
n = -2
try: 
    x = factorial(n)
except ValueError:
    x = -1 # default value
    print('There was a problem with the factorial.')
# . . .
print('We have landed!')


# %%
