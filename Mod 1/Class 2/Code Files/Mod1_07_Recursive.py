#%%
# recursive functions 
# essential skill for building tree type constructs
# watch out for stackoverflow and infinite loops
 
import math

#%%
# You can just as easily write this following function with loops
def triangleSum(n):
  """
  Finding the sum of 1+2+...+n
  Args:
      n (int): the last interger to be added
  Return: int
  """
  n = math.floor(n) # try to make sure it is an integer, not float
  if n > 1 :
    return n+triangleSum(n-1)
  return 1 # execute when n = 1

# Try the function
triangleSum(21)

#%%
# Try write a recursive function to calculate the "factorial". Example, 4-factorial = 4*3*2*1 returns 24
def factorial(n):
  """
  Calculate the factorial n!
  Args:
      n (int): the last interger to be multiplied
  Return: int
  """
  n = math.floor(n) # try to make sure it is an integer, not float
  if (n>1):
    return n*factorial(n-1)
  return 1

# Try the function
factorial(4)

#%% [markdown]
# Now Let us try various debugging option in VS Code
# - Adding breakpoints
# - Conditional breakpoints
# - Hit Count breakpoints
# - Continue/Step-in/Step-out


#%% 
# If given an array of array of array of ...
nestedarray = [ 'Peter', ['Paul','Mary'], 'Steve', ['Charles',['Prince','Harry'],'Alexa',['Siri','Bigsby']]]
# Write a recursive function that counts how many names are all together (duplicate names count as 2)
def countNestedNames(names):
  """
  Count nested names, duplicate names are counted separately
  Args:
      names (list): the nested array of names
  Return: int
  """
  cnt = len(names)
  for item in names:
    if type(item)==list:
      #cnt = cnt-1
      cnt = cnt + countNestedNames(item)
      #cnt += countNestedNames(item)
    
  return cnt
countNestedNames(nestedarray)

#%%

# Let's work together on a challenging problem:
# The recusive tower of Hanoi
# Starting with tower A having disks of sizes 1,2,3 stacked, 
# our goal is to move them all to tower B, one disk at a time
# with tower C as a buffer. At any time, the disks on each tower 
# cannot have larger disks above smaller disks. Show the steps of 
# how to do move them.


#%%
import math

# this is the front-end, to take inputs, initialize variables, integrity checks
def movetowers(levels, fromt=0, tot=1) :
  """
  Moving stacks of blocks from one tower to another, showing the detail steps
  Args:
      levels (int): the number of levels to move
      fromt (int): the id of the fromTower 
      tot (int): the the id of the toTower 
  Return: None
  """
  # build the tower shell
  towers = [ { "id":0,"name":'A',"stack":[] } , { "id":1,"name":'B',"stack":[] } , { "id":2,"name":'C',"stack":[] } ]
  for i,t in enumerate(towers) :
    t['stack'] = list(range(1,levels+1)) if i==fromt else []
    # if i==fromt :
    #   t['stack']=list(range(1,levels+1))
    # else :
    #   t['stack']=[]

  # check input n
  levels = 2 if levels<1 else 50 if levels>50 else math.floor(levels)
  # if levels<1 :
  #   levels=2
  # elif levels>50 :
  #   levels=50
  # else :
  #   levels=math.floor(levels)
  
  # check input fromt (from tower id) should be 0,1, or 2
  fromt = 0 if fromt<0 else 2 if fromt >2 else math.floor(fromt)
  # if fromt < 0 :
  #   fromt = 0
  # elif fromt > 2 :
  #   fromt = 2
  # else :
  #   fromt = math.floor(fromt)
  
  # check input tot (to tower id) should be 0,1, or 2, AND not equal to fromt
  tot = 0 if tot<0 else 2 if tot>2 else math.floor(tot)
  # if tot < 0 :
  #   tot = 0
  # elif tot > 2 :
  #   tot = 2
  # else :
  #   tot = math.floor(tot)

  # make sure fromt != toto
  if tot == fromt :
    tot = (fromt+1)%3  # use of modular algebra (%-remainder) simplifies a lot of codes
  # tot = ((fromt+1)%3) if tot == fromt else tot
    
  print("At the begining:")
  localtowersdisp(towers)
      
  mvt(towers, levels, fromt, tot)
  return

# this is the back-end, bread and butter, function doing the heavy lifting. RECURSIVE
def mvt( towers, levels, fromt=0, tot=1):
  """
  Moving stacks of blocks from one tower to another, showing the detail steps
  Args:
      towers (list): the towers is a list consisting of the towers, and each tower is a dictionary
      levels (int): the number of levels to move
      fromt (int): the id of the fromTower 
      tot (int): the the id of the toTower 
  Return: None
  """
  dummyt = 3-fromt-tot
  
  if levels == 1 : # only 1 level to move, I know how to do it. Let's do it
    print("Move disk 1 from tower",towers[fromt]['name'],"to tower",towers[tot]['name'])
    towers[tot]['stack'].insert(0,towers[fromt]['stack'].pop(0))
    localtowersdisp(towers) # display the config after each move
    # return  # return None
  else : 
    # first move all top blocks except the bottom one to dummy tower
    mvt(towers, levels-1, fromt, dummyt) 
    
    # the actual move of the bottom block to the destination tower
    print("Move disk",levels,"from tower",towers[fromt]['name'],"to tower",towers[tot]['name'])
    towers[tot]['stack'].insert(0,towers[fromt]['stack'].pop(0))
    localtowersdisp(towers) # display the config after each move
    
    # move the dummy tower back to the destination tower
    mvt(towers, levels-1, dummyt, tot) 
    # return # return None
  return # return None

# Helper functions
def localtowersdisp(towers): 
  """
  Display blocks on each tower
  Args:
      towers (list): the towers is a list consisting of the towers, and each tower is a dictionary
  Return: None
  """
  print("Config- ", end =" ")
  for t in towers:
    print(str(t['name'])+": "+ str(t['stack'])+" ", end =" ")
    # print(t['name'], ":", t['stack'], end = " ")
  print() # linebreak
  return

#%%
# Now try it
# 
movetowers(3,2,0)
movetowers(4,2,0)

#%%
