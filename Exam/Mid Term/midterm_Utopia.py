# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'

#%%
# Basic packages here. 
# Feel free to import other packages if you need them.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns # if you (optionally) want to show fancy plots
import rfit 

world1 = rfit.dfapi('World1', 'id')
world1.to_csv("world1.csv")
# world1 = pd.read_csv("world1.csv", index_col="id") # use this instead of hitting the server if csv is on local
world2 = rfit.dfapi('World2', 'id')
world2.to_csv("world2.csv")
# world2 = pd.read_csv("world2.csv", index_col="id") # use this instead of hitting the server if csv is on local

print("\nReady to continue.")


#%%[markdown]
# # Two Worlds 
# 
# I was searching for utopia, and came to this conclusion: If you want to do it right, do it yourself. 
# So I created two worlds. 
#
# Data dictionary:
# * age00: the age at the time of creation. This is only the population from age 30-60.  
# * education: years of education they have had. Education assumed to have stopped. A static data column.  
# * marital: 0-never married, 1-married, 2-divorced, 3-widowed  
# * gender: 0-female, 1-male (for simplicity)  
# * ethnic: 0, 1, 2 (just made up)  
# * income00: annual income at the time of creation   
# * industry: (ordered with increasing average annual salary, according to govt data.)   
#   0. leisure n hospitality  
#   1. retail   
#   2. Education   
#   3. Health   
#   4. construction   
#   5. manufacturing   
#   6. professional n business   
#   7. finance   
# 
# 
# Please do whatever analysis you need, convince your audience both, one, or none of these 
# worlds is fair, or close to a utopia. 
# Use plots, maybe pivot tables, and statistical tests (optional), whatever you deem appropriate 
# and convincing, to draw your conclusions. 
# 
# There are no must-dos, should-dos, cannot-dos. The more convincing your analysis, 
# the higher the grade. It's an art.
#

#%%