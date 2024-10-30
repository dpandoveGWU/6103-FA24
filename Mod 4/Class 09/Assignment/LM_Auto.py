#This question involves the use of multiple linear regression on the Auto data set from ISLR library
#(a) Produce a scatterplot matrix which includes all of the variables  in the data set.  
# (b) Compute the matrix of correlations between the variables using  the DataFrame.corr() method.  
# (c) Use the sm.OLS() function to perform a multiple linear regression  with mpg as the response and all other variables except name as the predictors. Use the summarize() function to print the results. 
#%%
from ISLP import load_data
Auto = load_data('Auto')
Auto.columns

# %%
