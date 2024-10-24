#%%
# print("Hello world!")


#%%
# Question 1: Create a Markdown cell with the followings:
# Two paragraphs about yourself. In one of the paragraphs, give a hyperlink of a website 
# that you want us to see. Can be about yourself, or something you like.

#%%[markdown]
#
# This is [my name]. 
# I am a student of xxxx
#
# Second paragraph here. And I like to [code](https://en.wikipedia.org/wiki/The_Da_Vinci_Code_(film)).


#%%
# Question 2: Create
# a list of all the class titles that you are planning to take in the data science program. 
# Have at least 6 classes, even if you are not a DS major
# Then print out the last entry in your list.

myCourseTitles = ['Intro DS', 'Intro DM', 'Data Warehousing', 'NLP', 'Capstone', 'Time Series Analysis']
print(myCourseTitles[-1])

#%%
# Question 3: After you completed question 2, you feel Intro to data mining is too stupid, so you are going 
# to replace it with Intro to Coal mining. Do that in python here.

myCourseTitles[1] = 'Coal Mining'

#%%
# Question 4: Before you go see your acadmic advisor, you are 
# asked to create a python dictionary of the classes you plan to take, 
# with the course number as key. Please do that. Don't forget that your advisor 
# probably doesn't like coal. And that coal mining class doesn't even have a 
# course number.

myCourseList = { 6101:'Intro DS', 6103:'Intro DM', 6102:'Data Warehousing', 6312:'NLP', 6501:'Capstone', 6450:'Time Series Analysis'}

#%%
# Question 5: print out and show your advisor how many 
# classes (print out the number, not the list/dictionary) you plan 
# to take.
print(len(myCourseList))

#%%
# Question 6: Using loops 
# Goal: print out the list of days (31) in Jan 2021 like this
# Sat - 2022/1/1
# Sun - 2022/1/2
# Mon - 2022/1/3
# Tue - 2022/1/4
# Wed - 2022/1/5
# Thu - 2022/1/6
# Fri - 2022/1/7
# Sat - 2022/1/8
# Sun - 2022/1/9
# Mon - 2022/1/10
# Tue - 2022/1/11
# Wed - 2022/1/12
# Thu - 2022/1/13
# ...
# You might find something like this useful, especially if you use the remainder property x%7
# dayofweektuple = ('Sun','Mon','Tue','Wed','Thu','Fri','Sat') # day-of-week-tuple

dayofweektuple = ('Sun','Mon','Tue','Wed','Thu','Fri','Sat')
for i in range(31):
  print(dayofweektuple[(i+6)%7], '-', '2022/1/'+str(i+1) )


# %%[markdown]
# # Additional Exercise: 
# Choose three of the five exercises below to complete.
#%%
# =================================================================
# Class_Ex1: 
# Write python codes that converts seconds, say 257364 seconds,  to 
# (x Hour, x min, x seconds)
# ----------------------------------------------------------------

x=257364
print(f"({x//3600} Hour, {(x-3600*(x//3600))//60 } min, {x%60} seconds)")




#%%
# =================================================================
# Class_Ex2: 
# Write a python codes to print all the different arrangements of the
# letters A, B, and C. Each string printed is a permutation of ABC.
# Hint: one way is to create three nested loops.
# ----------------------------------------------------------------

# Need to create three positions, and loop them through, 
# and check to avoid duplicates
phrase = 'ABC'
for p1 in phrase:
  for p2 in phrase:
    for p3 in phrase:
      if (p1 != p2 and p2 != p3 and p3 != p1): print(p1+p2+p3)
      
#%%
# alternative style/logic
phrase = 'ABC'
for p1 in phrase:
  for p2 in phrase:
    for p3 in phrase:
      if ( len({p1,p2,p3}) < len(phrase) ) : continue # This will be much shorter than previous when you have more than three loops
      print(p1+p2+p3)




#%%
# =================================================================
# Class_Ex3: 
# Write a python codes to print all the different arrangements of the
# letters A, B, C and D. Each string printed is a permutation of ABCD.
# ----------------------------------------------------------------

phrase = 'ABCD'
for p1 in phrase:
  for p2 in phrase:
    for p3 in phrase:
      for p4 in phrase:
        if ( len({p1,p2,p3,p4}) < len(phrase) ) : continue # This will be much shorter than previous when you have more than three loops
        print(p1+p2+p3+p4)



#%%
# =================================================================
# Class_Ex4: 
# Suppose we wish to draw a triangular tree, and its height is provided 
# by the user, like this, for a height of 5:
#      *
#     ***
#    *****
#   *******
#  *********
# ----------------------------------------------------------------

# A quick inspection finds that for n=5, a single asterisk at position n (or index n-1 if you want), 
# then we need 3 asterisks starting at position n-1, ...
# and finally, 2n+1 asterisks, starting from position 1, (or index 0)
n = int(input("How many levels (n)?"))
for i in range(n):
  print(' '*(n-i-1) + '*'*(2*i+1))



#%%
# =================================================================
# Class_Ex5: 
# Write python codes to print prime numbers up to a specified 
# values, say up to 200.
# ----------------------------------------------------------------

# plan: we can loop through all values from 1 to nmax of 200
# test each number see if it is prime. 
# There are better and more efficient ways; for now, we just want 
# to make sure it works.

# One thing to note: when checking if a number is prime, we can test if 
# the number is divisible by a smaller numer. We only need to check the 
# divisibility up to square root of the number. If you did not 
# find any number being a factor, up to sqrt of n, you will not find one 
# above sqrt n. 
 
nmax = 200; print(2) # 2 is a special case
for i in range(nmax+1): 
  for testval in range(2,i):
    if i%testval == 0: break # testval is a factor, try next i value, break from this inner loop
    if testval*testval < i: continue # testval is still small enough, keep trying this inner loop.
    print(i) # This passed all previous tests, never "break", so this must be a prime.
    break # At first, I didn't notice I need this break. As a result, the loop continues although the sqrt value has been reached. Repeated prime values will be printed out as a result without this break





# =================================================================
# %%
