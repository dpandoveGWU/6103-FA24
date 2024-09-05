# %%
# Control structure
# Use `if` to control the flow of a program based on a condition
# Usage: 
# if <condition>:
#   <statement 1> # mind the indentation
# else:
#   <statement 2>
# <condition>: a boolean (True / False)


# %%[markdown]
# ### Relational operators: <expr> <relop> <expr>
#
#| Python   | Mathematics | Meaning                  |
#| -------- | ----------- | ------------------------ |
#| <        | $<$         | less than                |
#| <=       | $\le$       | less than or equal to    |
#| ==       | $=$         | equal to                 |
#| >=       | $\ge$       | greater than or equal to |
#| >        | $>$         | greater than             |
#| !=       | $\neq$      | not equal                |

# %%
# try the following lines different operators
print( 3 >= 4 )
print( 'Hello world' == "Hello world" )
print( 0.1 + 0.1 == 0.2 )


# %%
# String less or greater than
# take a guess
print('Hello world' <= 'Hello' )


# %%
# now thy this
0.1 + 0.1 + 0.1 == 0.3
# Is Python crazy?


# %%
# Even or odd??
x = input('Enter an integer: ') 
# Check if even or odd
if x % 2 == 0:
  print('Even')
else:
  print('Odd')


# %%
# If-else shorthand
x = 101
print('Even') if x % 2 == 0 else print('Odd')

# %%
# Activity - 1
# Nesting loops and conditionals
# Write a code that checks if an integer is a prime number.
x = 101
isPrime = True
for i in range(2, x):
    if x % i == 0:
        isPrime = False

print('Prime') if isPrime else print('Not prime')
#
#


# %%
# Activity - 3
# Nested conditional
# Check if a given year is a leap year
# Rule 1: the year must be divisible by 4
# Rule 2: if the year is divisible by 100, then must also be divisible by 400
def isLeapYear(year):
    if year < 1:
        print('Enter a positive year')
        return 
    return (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0)
 
# 
# 
#  


# %%
# Two-way decisions: if elif elif ....
# 
# if <condition 1>:
#  <statement 1>
# elif <condition 2>:
#   <statement 2>
# . . . 
# else :
#  <the last statement>

# %%
# Number to letter grade
grade = 50
if grade >= 93:
    print('A')
elif grade >= 90:
    print('A-')
elif grade >= 87:
    print('B+')
elif grade >= 83:
    print('B')
elif grade >= 80:
    print('B-')
elif grade >= 77:
    print('C+')
elif grade >= 73:
    print('C')
elif grade >= 70:
    print('C-')
else:
    print('F')