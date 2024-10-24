###############  HW  Functions      HW  Functions         HW  Functions       ###############
#%%
# ######  QUESTION 1   First, review Looping    ##########
# Write python codes to print out the four academic years for a typical undergrad will spend here at GW. 
# Starts with Sept 2021, ending with May 2025 (total of 45 months), with printout like this:
# Sept 2021
# Oct 2021
# Nov 2021
# ...
# ...
# Apr 2025
# May 2025
# This might be helpful:
# If you consider Sept 2021 as a number 2021 + 8/12, you can continue to loop the increament easily 
# and get the desired year and month. (If the system messes up a month or two because of rounding, 
# that's okay for this exercise).
# And use this (copy and paste) 
# monthofyear = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec')
# to simplify your codes.


monthofyear = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec')

for i in range(45):
    print( monthofyear[(i+8)%12], 2021+int((i+8)/12) )
    

###############  Now:     Functions          Functions             Functions       ###############
# We will now continue to complete the grade record that we were working on in class.

#%%
###################################### Question 2 ###############################
# let us write a function find_grade(total)  
# which will take your course total (0-100), and output the letter grade (see your syllabus)
# have a habbit of putting in the docstring
total = 62.1

def find_grade(total):
  # write an appropriate and helpful docstring
  """
  convert total score into grades
  :param total: 0-100 
  :return: str
  """
  grade = 0 # initial placeholder
  # use conditional statement to set the correct grade
  grade = 'A' if total>=93 else 'A-' if total >= 90 else 'B+' if total >= 87 else 'B' if total >= 83 else 'B-' if total >=80 else 'C+' if total >= 77 else 'C' if total >= 73 else 'C-' if total >=70 else 'D' if total >=60 else 'F' 
  return grade  

# Try:
print(find_grade(total))

# Also answer these: 
# What is the input (function argument) data type for total? 
# integer or float
# What is the output (function return) data type for find_grade(total) ?
# string


#%%
###################################### Question 3 ###############################
# next the function to_gradepoint(grade)
# which convert a letter grade to a grade point. A is 4.0, A- is 3.7, etc
grade = 'C-'

def to_gradepoint(grade):
  # write an appropriate and helpful docstring
  """
  convert letter grades into gradepoint
  :param str grade: A, A-, B+, B, B-, C+, C, C-, D, F
  :return: float
  """
  gradepoint = 0 # initial placeholder
  # use conditional statement to set the correct gradepoint
  gradepoint = 4 if grade=='A' else 3.7 if grade=="A-" else 3.3 if grade=="B+" else 3 if grade=="B" else 2.7 if grade=="B-" else 2.3 if grade=="C+" else 2 if grade=="C" else 1.7 if grade=='C-' else 1 if grade=="D" else 0 
  return gradepoint

# Try:
print(to_gradepoint(grade))

# What is the input (function argument) data type for to_gradepoint? 
# string
# What is the output (function return) data type for to_gradepoint(grade) ?
# float


#%%
###################################### Question 4 ###############################
# next the function to_gradepoint_credit(course)
# which calculates the total weight grade points you earned in one course. Say A- with 3 credits, that's 11.1 total grade_point_credit
course = { "class":"IntroDS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 

def to_gradepoint_credit(course):
  # write an appropriate and helpful docstring
  """
  calculate the gradePoint credit from a course-record of a student
  :param course: a dictionary with keys: grade, credits, class, semester, year, class, id
  :return: float
  """
  grade_point_credit = course['credits']*to_gradepoint(course['grade'])
  # eventually, if you need to print out the value to 2 decimal, you can 
  # try something like this for floating point values %f
  # print(" %.2f " % grade_point_credit)
  return grade_point_credit

# Try:
print(" %.2f " % to_gradepoint_credit(course) )

# What is the input (function argument) data type for to_gradepoint_credit? 
# dictionary
# What is the output (function return) data type for to_gradepoint_credit(course) ?
# float


#%%
###################################### Question 5 ###############################
# next the function gpa(courses) to calculate the GPA 
# It is acceptable syntax for list, dictionary, JSON and the likes to be spread over multiple lines.
courses = [ 
  { "class":"Intro to DS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } , 
  { "class":"Data Warehousing", "id":"DATS 6102", "semester":"fall", "year":2018, "grade":'A-', "credits":4 } , 
  { "class":"Intro Data Mining", "id":"DATS 6103", "semester":"spring", "year":2018, "grade":'A', "credits":3 } ,
  { "class":"Machine Learning I", "id":"DATS 6202", "semester":"fall", "year":2018, "grade":'B+', "credits":4 } , 
  { "class":"Machine Learning II", "id":"DATS 6203", "semester":"spring", "year":2019, "grade":'A-', "credits":4 } , 
  { "class":"Visualization", "id":"DATS 6401", "semester":"spring", "year":2019, "grade":'C+', "credits":3 } , 
  { "class":"Capstone", "id":"DATS 6101", "semester":"fall", "year":2019, "grade":'A-', "credits":3 } 
  ]

def find_gpa(courses):
  # write an appropriate and helpful docstring
  """
  calculate the grade-point-average from a list of course-records of a student
  :param courses: a list of courseRecords, each with keys: grade, credits, class, semester, year, class, id
  :return: float
  """
  total_grade_point_credit = 0 # initialize cumulative grade-point total
  total_credits = 0 # initialize cumulative credits total
  for course in courses:
    total_grade_point_credit += to_gradepoint_credit(course)
    # total_grade_point_credit = total_grade_point_credit + to_gradepoint_credit(course)
    total_credits += course['credits']
    # total_credits = total_credits + course['credits']
  
  gpa = total_grade_point_credit/total_credits
  return gpa

# Try:
print(" %.2f " % find_gpa(courses) )

# What is the input (function argument) data type for find_gpa? 
# What is the output (function return) data type for find_gpa(courses) ?


#%%
###################################### Question 6 ###############################
# Write a function to print out a grade record for a single class. 
# The return statement for such functions should be None or just blank
# while during the function call, it will display the print.
course = { "class":"IntroDS", "id":"DATS 6101", "semester":"spring", "year":2018, "grade":'B-', "credits":3 } 

def printCourseRecord(course):
  # write an appropriate and helpful docstring
  # use a single print() statement to print out a line of info as shown here
  # 2018 spring - DATS 6101 : Intro to DS (3 credits) B-  Grade point credits: 8.10 
  """
  print line for a course-record of a student
  :param course: a dictionary with keys: class, semester, year, grade, credits 
  :return: None
  """
  print("%d %s - %s : %s (%d credits) %s Grade point credits: %.2f" % ( course['year'], course['semester'], course['id'], course['class'], course['credits'] , course['grade'], to_gradepoint_credit(course) ) )
  return # or return None

# Try:
printCourseRecord(course)

# What is the input (function argument) data type for printCourseRecord? 
# dictionary
# What is the output (function return) data type for printCourseRecord(course) ?
# None


#%%
###################################### Question 7 ###############################
# write a function (with arguement of courses) to print out the complete transcript and the gpa at the end
# 2018 spring - DATS 6101 : Intro to DS (3 credits) B-  Grade point credits: 8.10 
# 2018 fall - DATS 6102 : Data Warehousing (4 credits) A-  Grade point credits: 14.80 
# ........  few more lines
# Cumulative GPA: ?????
 
def printTranscript(courses):
  # write an appropriate and helpful docstring
  for course in courses:
    printCourseRecord(course)
  
  # after the completion of the loop, print out a new line with the gpa info
  print("Cumulative GPA: %.2f" % find_gpa(courses))
  
  return # or return None

# Try to run, see if it works as expected to produce the desired result
# courses is already definted in Q4
printTranscript(courses)

# What is the input (function argument) data type for printTranscript? 
# list of dictionary
# What is the output (function return) data type for printTranscript(courses) ?
# None



#%% 
# ######  QUESTION 8   Recursive function   ##########
# Write a recursive function that calculates the Fibonancci sequence.
# The recusive relation is fib(n) = fib(n-1) + fib(n-2), 
# and the typically choice of seed values are fib(0) = 0, fib(1) = 1. 
# From there, we can build fib(2) and onwards to be 
# fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5, fib(6)=8, fib(7)=13, ...
# Let's set it up from here:

def fib(n):
  """
  Finding the Fibonacci sequence with seeds of 0 and 1
  The sequence is 0,1,1,2,3,5,8,13,..., where 
  the recursive relation is fib(n) = fib(n-1) + fib(n-2)
  :param n: the index, starting from 0
  :return: the sequence
  """
  n = n//1 if n>=1 else 0 # ensure n is non-negative integer
  if n>1:
    return fib(n-1) + fib(n-2)

  # n should be <= 1 here. Anything greater than 0, assume it's 1
  return 1 if (n==1) else 0 # seed values fib(1) = 1, fib(0) = 0 
  # elif n==0:   # n should be <= 1 here. Anything greater than 0, assume it's 1
    # return 1  # this is fib(1) seed value
  # else:
    # return 0  # this is fib(0) seed value


# Try:
for i in range(12):
  print(fib(i))  



#%% 
# ######  QUESTION 9   Recursive function   ##########
# Similar to the Fibonancci sequence, let us create one (say dm_fibonancci) that has a  
# modified recusive relation dm_fibonancci(n) = dm_fibonancci(n-1) + 2* dm_fibonancci(n-2) - dm_fibonancci(n-3). 
# Pay attention to the coefficients and their signs. 
# And let us choose the seed values to be dm_fibonancci(0) = 1, dm_fibonancci(1) = 1, dm_fibonancci(2) = 2. 
# From there, we can build dm_fibonancci(3) and onwards to be 1,1,2,3,6,10,...
# Let's set it up from here:

def dm_fibonancci(n):
  """
  Finding the dm_Fibonacci sequence with seeds of 1, 1, 2 for n = 0, 1, 2 respectively
  The sequence is 0,1,1,2,3,5,8,13,..., where 
  the recursive relation is dm_fibonancci(n) = dm_fibonancci(n-1) + 2* dm_fibonancci(n-2) - dm_fibonancci(n-3)
  :param n: the index, starting from 0
  :return: the sequence
  """
  # assume n is positive integer
  n = n//1 if n>=1 else 0 # ensure n is non-negative integer
  c1=1
  c2=2
  c3=-1
  if n>2:
    return c1* dm_fibonancci(n-1) + c2* dm_fibonancci(n-2) + c3* dm_fibonancci(n-3)

  # n should be <= 1 here. Anything greater than 0, assume it's 1
  return 2 if (n==2) else 1 if (n==1) else 1 # seed values dm_fibonancci(2) = 2, dm_fibonancci(1) = 1, dm_fibonancci(0) = 1 

for i in range(12):
  print(dm_fibonancci(i))  # should gives 1,1,2,3,6,10,...


#%%

