#%%
import math
import os

print("Hello world!")

#%%
# function
# good habit to have (at least) a return statement for any functions
# python use indentations instead of (curly) brackets
def my_add(a=0, b=0):
  my_sum = a+b
  # print(my_sum)
  return my_sum
  # return 


#%%
# Include docstring whenever possible
def my_times(a=1, b=1):
  """
  multiplying two floats
  Args:
      a (float): any number
      b (float): any number
  Return: float
  """
  my_product = a*b
  print(my_product)
  return my_product
  # return

#%%
santa = 'ho'
# Question: What is the difference these two ways of doing things:
# first
def myfunction(a):
  out = a+a+a
  print(out)

myfunction(santa)

# second way
def myfunction2(a):
  out = a+a+a
  return out

print(myfunction2(santa))

# Answer: second method is better for many reasons...
# Do you see the function as a "user-interface" function that interact with the user, or 
# do you see it as a tripling function, that deals with data and values? 
# GIVE functions a return value, give them a purpose, a clear reason of their existence.


#%%
# Let us write a function to find the Greatest Common Divisor between two integers
def gcd(num1, num2):
  """
  find the Greatest Common Divisor between two integers
  Args:
      num1 (int): any positive integer
      num2 (int): any positive integer
  Return: int
  """
  min = num1 if num1 < num2 else num2
  largestFactor = 1
  for i in range(1, min + 1):
      if num1 % i == 0 and num2 % i == 0:
          largestFactor = i
  return largestFactor

# Now we can try use the function
L = gcd(24, 6)
print(L)
print('#',50*"-")

# Can you think of better ways to implement this gcd function?
# We can brainstorm and write some pseudocode...

#%%
# let us write a function find_grade(total) 
# which will take your course total (0-100), and output the letter grade (see your syllabus)
# have a habbit of putting in the docstring
total = 82.74

#%%
# Let us write a function to_gradepoint(grade)
# which convert a letter grade to a grade point. A is 4.0, A- is 3.7, etc
grade = 'B-'

#%%
# Next write a function to_gradepoint_credit(course)
# which calclates the total weight grade points you earned in one course. Say A- with 3 credits, that's 11.1 total grade_point_credit
course = { "class":"IntroDS", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 

#%%
# Now write a function gpa(courses) to calculate the GPA 
courses = [ { "class":"IntroDS", "semester":"spring", "year":2018, "grade":'B-', "credits":4 } , { "class":"IntroDataMining", "semester":"fall", "year":2018, "grade":'A', "credits":3 } ]

#%%
# global variables vs local scope
greet = "Great!"
thismonth = 'Sept'
def santaSays(s):
  """ Echo what Santa says """
  global thismonth
  greet = "It's already Dec! " if (thismonth=='Dec' or thismonth=="December") else "It is only "+thismonth+". "
  return greet + 3*s if (s=="ho" or s=="Ho" or s=="HO") else greet + s

# %%
print(santaSays('oh'))
print(santaSays('ho'))
thismonth = "Dec"
print(santaSays('oh'))
print(santaSays('ho'))

# %%
