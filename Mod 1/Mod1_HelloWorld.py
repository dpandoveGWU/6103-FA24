#%%
# This is a comment line
# The first line #%% is a "magic puython" command. Some interpreters knows what it is, others just ignore them.
# In our case here, VSCode knows #%% is the beginning of a cell, like Jupyter notebook cells
# We can also specify it as a Markdown cell (see bottom of this file) with #%%[markdown] 
print("Hello world")
print('Basic Arithmetics')
print(5 / 8)
print (7+10)
print(10/3, 3/10) # print numerical divisions
print(10//3, 3//10) # print quotients from divisions
print(10%3, 3%10) # print remainders from divisions,

#%%
astring = "Thank you" 
# #######################################################
# In python, 'single' quotes and "double" quotes are the same, as long as you use the same for opening and closing.
# Note: `backtick` is different. Not the same as 'single'/"double" quotes
# #######################################################

anum = 3.14159265358979323846
cnt = 1

# The above three definitions can also be combined on one-line if you like:
string, anum, cnt = 'Thank you', 3.14159265358979323846, 1

# Many different ways to print out the same line
print("%d. I want to say %s" % (cnt,astring) )
cnt+=1
print(cnt,". I want to say" + astring )
cnt+=1
print(cnt, ". I want to say",astring )
cnt+=1
print("%d. I want to say %s, my sweetie %.3f" % (cnt,astring,anum) )
cnt+=1
print("%d. I want to say %s, my sweetie digit %d" % (cnt,astring,anum) )
cnt+=1
print("%d. I want to say %s, my sweetie long %f" % (cnt,astring,anum) )
cnt+=1
# #######################################################
# For python 3.6+, we can use the f-string 
# Use this often!! Avoid the old methods/formats 
# Much easier to control the output this way. 
# And a lot of trouble have been taken care of for you.
# ########################################################
print(f"{cnt}. I want to say {astring}, my sweetie long {anum.__round__(3)}")
cnt+=1

#%%
# Seeing/trying is believing. Try the followings:
#  
print(f'This is how I like my {anum}')
print( 'This is how I like my '+ str(anum) )
print( 'This is how I like my '+ anum )

#%%
# Also try this... (we'll formally introduce list/array in the next section)
alist = [3,'apples',5,'oranges']
print(f'This is how I like my {alist}')
print( 'This is how I like my ' + str(alist) )
print( 'This is how I like my ' + alist.__repr__()  )
print( 'This is how I like my ' + alist )

#%%
# see https://python-reference.readthedocs.io/en/latest/docs/str/formatting.html
# s-string, d-digit (int), f-float
# 
# side note:  
# for more info on the new python f-string since python 3.6, see
# for example: https://cito.github.io/blog/f-strings/
#
# also, see https://docs.python.org/3/reference/lexical_analysis.html#literals
# for info on f-string, r-string, b-string, etc.


#%%[markdown]
#
# # Python Class 01
#
# This is our first class of the semester.
# Hello to everyone.   
# Two spaces in the previous line doesn't make a new line in this environment. 
#
# You will need a blank line to get a new paragraph.

# The above is not considered a blank line without the # sign.
#
# This can get you a [link](http://www.gwu.edu).
#
# You can find some cheatsheets to do other basic stuff like bold-face, italicize, tables, etc.

# %%
# WHY NOT Jupyter Notebook ipynb ?????
# 
# Not git friendly. 
# Jupyter notebooks are saved in a JSON format, and typically as a sinble line!
# Let's try.
#
# Notice that we can easily open ipynb files in VSCode (with the proper extensions installed) 
# And we can easily export any of your .py file and results as ipynb as well. 
# Unless you are going solo on all your work, there very little reson to use Jupyter notebook.
