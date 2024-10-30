# %%
from wooldridge import dataWoo
import pandas as pd
import numpy as np
import rfit
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

# %%
wage1 = dataWoo('wage1')

# %%
# A curious question is the association of `wage` and `educ`
sns.scatterplot(x = 'educ', y = 'wage', hue = 'female',  data = wage1)
plt.xlabel("educ in years")
plt.ylabel("hourly wage in USD")
plt.title("Educ vs Wage")
plt.show()

# %%
wage1["log_wage"] = np.log(wage1['wage'])

# %%
sns.scatterplot(x = 'educ', y = 'log_wage', hue = 'female',  data = wage1)
plt.xlabel("educ in years")
plt.ylabel("log(hourly wage) in USD")
plt.title("Educ vs Log Wage")
plt.show()


# %%
# Plot the OLS line (without and with hue)
sns.lmplot(x = 'educ', y = 'log_wage', data=wage1, fit_reg = True, ci = None, scatter_kws={'alpha': 0.4, 's': 8 })

# %%
# Build a linear model
modelEdWage = ols(formula = 'log_wage ~ educ', data = wage1)
print( type(modelEdWage) )


# %%
# Fit and display the summary
modelEdWageFit = modelEdWage.fit()
print( modelEdWageFit.summary() )

# %%
# How to interpret the coeff and std. err.


# %%
# Let's now answer the research question
model = ols(formula = 'log_wage ~ educ + C(female)', data = wage1)
model = model.fit()
print( model.summary() )


# %%
# A better model
modelEdTenureWage = ols(formula = 'log_wage ~ educ + tenure', data = wage1)
modelEdTenureWageFit = modelEdTenureWage.fit()
print( modelEdTenureWageFit.summary() )

