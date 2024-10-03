#%%
# Recursion
# Same similarity
# essential skill for building tree type constructs
# watch out for stackoverflow and infinite loops
# %%
import random
import time


# %%
# Recall our factorial function
def factorial(n):
  prod = 1
  for i in range(1, n + 1):
    prod *= i
  return prod

factorial(5)



# %%
# Calling a function by itself
# Recursive formula: n! = n * (n - 1)! for n => 0
def recFactorial(n):
  # Base case
  if n <= 1:
    return 1
  # Recursive step
  return n * recFactorial(n - 1)

recFactorial(5)


# %%
start_time = time.time()
recFactorial(1000)
print('Time taken:', (time.time() - start_time ) *10**3 )

start_time = time.time()
factorial(1000)
print('Time taken:', (time.time() - start_time ) *10**3 )



# %%
# Activity: 2
# Write recursive function to get the sum of the first n natural numbers.
# Formula: S_n = n + S_{n-1}
def recSum(n):
  if n == 1:
    return 1
  
  return n + recSum(n - 1)

print( recSum(5) )


# %%
# Activity: 3
# Write recursive function to compute the gcd of two integers
# Formula: gcd(a, b) = gcd(b, a % b)
def recGCD(a, b):
  # Base case
  if b == 0:
    return a

  # Recursive step
  return recGCD(b, a % b)

print( recGCD(16, 24) )



# %%
# Activity: 4
# Write recursive function to get the n-th Fibonacci number.
# Formula: F_n = F_{n - 1} + F_{n - 1}
def recFib(n):
  # Base case(s)
  if n == 1:
    return 0
  if n == 2:
    return 1
  # Recursive step
  return recFib(n - 1) + recFib(n - 2)

print( recFib(6) )
# Why is this much slower than our usual `fib` function?
# Ans: 


# %%
# Activity: 6a
# Permutations
def perm(word):
  # Base step
  if len(word) <= 1:
    return [word]

  # Recursive step
  for i in range(0, len(word)):
    perm(word[0:i])
    perm(word[i + 1:len(word)]) 


print( perm('ab') )

# %%
# Activity: 6b
# Think how to do it without recursion



# %%
# Activity: 5
# List reversal
# Formula: 
def recReverse(list = [], start = 0):
  # Base step
  end = len(list) - start - 1
  if start >= end:
    return
  # Recursive step
  # Swap start and end
  list[start], list[end] = list[end], list[start]
  start += 1
  recReverse(list, start)
  return list

list = [0, 1, 2, 3]
print( recReverse(list.copy()) )
print( list )

