# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%%[markdown]'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #optional

world1 = pd.read_csv("world1.csv", index_col="id")
world2 = pd.read_csv("world2.csv", index_col="id") 

print("\nReady to continue.")


# %%[markdown]
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
# There are no must-dos, should-dos, cannot-dos. The more convenicing your analysis, 
# the higher the grade. It's an art.
#

# %%
# getting to know the data
print("Sample world1 data",world1.head())
print("\n Sample world2 data",world2.head())

# %%
print(world1.info())
print(world2.info())
# %%
# Summary statistics for World 1
summary1 = world1.describe()
print("Summary statistics for World 1:\n", summary1)

# Summary statistics for World 2
summary2 = world2.describe()
print("\n Summary statistics for World 2:\n", summary2)
# %%
def summarize_data(data, world_name):
    summary = data.describe()
    print(f"\nSummary statistics for {world_name}:\n", summary)
    return summary

def compare_statistics(summary1, summary2):
    comparison = summary1.compare(summary2)
    print("\nComparison of Summary Statistics:\n", comparison)
    return comparison

# Load the datasets
world1 = pd.read_csv("world1.csv", index_col="id")
world2 = pd.read_csv("world2.csv", index_col="id")

# Summarize data
summary1 = summarize_data(world1, "World 1")
summary2 = summarize_data(world2, "World 2")

# Compare statistics
comparison = compare_statistics(summary1, summary2)
comparison

# The basic summary statistics shows nearly identical on all the metrics for both world1 and world2.
# The set up is nearly identical and will analyse using various methods

# %%[markdown]

# In an ideal world (utopia), we would expect certain conditions to be true, such as:
# * Equity in income: A more equal distribution of income with fewer disparities.
# * High education levels: Higher education levels across the population.
# * Balanced gender representation: No significant bias in gender representation.
# * Social and ethnic diversity: No significant ethnic or social marginalization.
# Hypothesis
# * Hypothesis 1: The two worlds significantly differ in income distribution, education, and social conditions (such as gender and ethnic representation).
    #    * I want to assess whether there are major differences between the worlds in terms of income equality, levels of education, and social diversity across gender and ethnicity.
# * Hypothesis 2: One world demonstrates more utopian characteristics, specifically more income equality, higher levels of education, and greater social balance (gender and ethnic equity)
   #    * I am going to see if one world is closer to an ideal or utopian world by analyzing how balanced and fair the distribution of income, education, gender, and ethnicity is.

# %%
# Correlation matrix for World 1
plt.figure(figsize=(12, 8))
sns.heatmap(world1.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for World 1")
plt.show()

# Correlation matrix for World 2
plt.figure(figsize=(12, 8))
sns.heatmap(world2.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for World 2")
plt.show()

# From the correlation matrix I can see that strong correlation between Income and Industry, 
# Week Positive correlation between Ethnicity and Income,and week positive correlation between gender and income ,
# and no correlation and negative correlation Age&Income,Marital Status&Income, Education&Income respectively for world1
# There is also strong Positive correlation between Industry and Income for world2 data
# I will will consider these correlation matrix results for further analysis

#%%
# Income Distribution by Gender and Industry
plt.figure(figsize=(15, 8))

# World 1: Boxplot of income by gender and industry
plt.subplot(1, 2, 1)
sns.boxplot(x='industry', y='income00', hue='gender', data=world1)
plt.title('World 1: Income Distribution by Gender and Industry')
plt.xlabel('Industry')
plt.ylabel('Income')

# World 2: Boxplot of income by gender and industry
plt.subplot(1, 2, 2)
sns.boxplot(x='industry', y='income00', hue='gender', data=world2)
plt.title('World 2: Income Distribution by Gender and Industry')
plt.xlabel('Industry')
plt.ylabel('Income')

plt.tight_layout()
plt.show()

# from the box plot of income by gender and industry Income increases with higher industry numbers for both world1 and world2.
# The overall pattern shows that industries with higher codes (like Professional & Business and Finance) tend to pay more.
# There is Noticeable gender disparity in Manufacturing, Professional and Business, particularly in Finance, where males earn substantially more for world1.
# There is also Higher incomes, with noticeable disparities in Manufacturing , Professional and Business, and Finance Industry of world2. 
# However, the gender income gap in Finance is less pronounced compared to World 1.
                 
                            # Hypothesis 1:
# The two worlds do differ significantly in terms of income equality and gender representation.
# World 1 has greater disparities in income, both within industries and between genders.
# World 2 shows less income inequality, especially in higher-income industries, and more gender balance.
# Therefore, Hypothesis 1 is supported: there are clear differences between the two worlds in terms of economic equality and social fairness.
                    # Hypothesis 2:
# World 2 demonstrates more utopian characteristics:
# It has a more equal distribution of income.
# It shows less disparity between genders, particularly in higher-paying jobs.
# This suggests World 2 is closer to a utopia because it aligns more with the ideals of equality and fairness across income and gender.

# %%

plt.figure(figsize=(14, 6))
# World 1: Boxplot of income by ethnicity
plt.subplot(1, 2, 1)  # First plot on the left
sns.boxplot(x='ethnic', y='income00', data=world1, palette="Blues")
plt.title('World 1: Income Distribution by Ethnicity', fontsize=14)
plt.xlabel('Ethnic Group', fontsize=12)
plt.ylabel('Income', fontsize=12)
plt.xticks([0, 1, 2], ['Ethnic Group 0', 'Ethnic Group 1', 'Ethnic Group 2'])

# World 2: Boxplot of income by ethnicity
plt.subplot(1, 2, 2)  
sns.boxplot(x='ethnic', y='income00', data=world2, palette="Oranges")
plt.title('World 2: Income Distribution by Ethnicity', fontsize=14)
plt.xlabel('Ethnic Group', fontsize=12)
plt.ylabel('Income', fontsize=12)
plt.xticks([0, 1, 2], ['Ethnic Group 0', 'Ethnic Group 1', 'Ethnic Group 2'])

plt.tight_layout()
plt.show()

                              # Hypothesis 1:
# from the above box plot1 World 1 shows a significant variation in income across different ethnic groups. 
# Ethnic Group 2, in particular, has a much wider income range compared to the other two groups. 
# This indicates notable disparities in income distribution across ethnicitie.
# from box plot2 world 2 , on the other hand, exhibits a more balanced income distribution across all ethnic groups. 
# The income range is narrower and more consistent, suggesting less income inequality among ethnic groups.
# Therefore, Hypothesis 1 is supported: The two worlds show distinct differences in income distribution by ethnicity, with World 2 being more equal and balanced.
                             #  Hypothesis 2:
# The more equal income distribution across ethnic groups in World 2 aligns with the ideals of a utopia, where income equality is more prevalent regardless of ethnic background. This suggests that World 2 fosters more social and economic fairness.
# In contrast, World 1 has greater income disparities. This shows that a society with more inequality, far from the utopian ideal of fairness.
# Therefore, Hypothesis 2 is supported: World 2 demonstrates more utopian characteristics by promoting income equality across different ethnic groups, while World 1 exhibits greater inequality.
# %%
# Education Level vs. Income

plt.figure(figsize=(14, 6))

# World 1: Scatter plot with regression line
plt.subplot(1, 2, 1)
sns.regplot(x='education', y='income00', data=world1, scatter_kws={'alpha':0.3})
plt.title('World 1: Education vs. Income')
plt.xlabel('Years of Education')
plt.ylabel('Income')

# World 2: Scatter plot with regression line
plt.subplot(1, 2, 2)
sns.regplot(x='education', y='income00', data=world2, scatter_kws={'alpha':0.3})
plt.title('World 2: Education vs. Income')
plt.xlabel('Years of Education')
plt.ylabel('Income')

plt.tight_layout()
plt.show()


                        # Hypothesis 1:
#  **World 1 scatter plot shows a broad spread of income across all levels of education. However, there is no strong upward trend between years of education and income. Some individuals with lower education still earn high incomes, and the overall distribution seems erratic, with many outliers, particularly at higher income levels.
# **World 2** (right plot) also exhibits a similar distribution, with minimal correlation between years of education and income. The spread remains fairly consistent across all education levels, and like World 1, some individuals with fewer years of education earn high incomes, suggesting little direct relationship between education and income.
# Therefore, **Hypothesis 1 is partially supported**: While both worlds show some variation in the relationship between education and income, neither world exhibits a strong correlation between higher education and higher income.

                        # Hypothesis 2:
# Both worlds show a **lack of strong connection** between years of education and income, which challenges the utopian ideal of meritocracy, where higher education should lead to higher income. In a utopian society, we would expect a clearer upward trend, with individuals receiving more rewards for higher education.
# Since both worlds fail to demonstrate this ideal relationship between education and income, **Hypothesis 2 is not fully supported**. Neither world appears to embody utopian characteristics in this regard, as education does not strongly influence income in either case.
# This analysis indicates that while other aspects of World 2 lean more towards utopia, the relationship between education and income does not align with the ideals of fairness and meritocracy in either world.

# %%
# Marital Status vs. Income

plt.figure(figsize=(14, 6))
# World 1: Boxplot of income by marital status
plt.subplot(1, 2, 1)
sns.boxplot(x='marital', y='income00', data=world1)
plt.title('World 1: Income Distribution by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Income')

# World 2: Boxplot of income by marital status
plt.subplot(1, 2, 2)
sns.boxplot(x='marital', y='income00', data=world2)
plt.title('World 2: Income Distribution by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Income')

plt.tight_layout()
plt.show()

# The income distribution by marital status shows consistent median incomes
# around $60,000 for all groups in both World 1 and World 2.
# This suggests that marital status does not significantly 
# impact income. Since utopia emphasizes fairness and equality, 
# this consistency implies that neither world allows marital status 
# to influence economic status unduly. Therefore, based on marital status 
# alone,neither world can be determined to be more utopian.

# %%
#Age vs. Income Analysis

plt.figure(figsize=(14, 6))

# World 1: Scatter plot with regression line for age vs. income
plt.subplot(1, 2, 1)
sns.regplot(x='age00', y='income00', data=world1, scatter_kws={'alpha':0.3})
plt.title('World 1: Age vs. Income')
plt.xlabel('Age')
plt.ylabel('Income')

# World 2: Scatter plot with regression line for age vs. income
plt.subplot(1, 2, 2)
sns.regplot(x='age00', y='income00', data=world2, scatter_kws={'alpha':0.3})
plt.title('World 2: Age vs. Income')
plt.xlabel('Age')
plt.ylabel('Income')

plt.tight_layout()
plt.show()



# %%
# Gender and Ethnicity Combinations

plt.figure(figsize=(14, 6))
# World 1: Bar plot of mean income by gender and ethnicity
plt.subplot(1, 2, 1)
world1_grouped = world1.groupby(['gender', 'ethnic'])['income00'].mean().unstack()
world1_grouped.plot(kind='bar', ax=plt.gca())
plt.title('World 1: Mean Income by Gender and Ethnicity')
plt.xlabel('Gender')
plt.ylabel('Mean Income')

# World 2: Bar plot of mean income by gender and ethnicity
plt.subplot(1, 2, 2)
world2_grouped = world2.groupby(['gender', 'ethnic'])['income00'].mean().unstack()
world2_grouped.plot(kind='bar', ax=plt.gca())
plt.title('World 2: Mean Income by Gender and Ethnicity')
plt.xlabel('Gender')
plt.ylabel('Mean Income')
plt.tight_layout()
plt.show()


# In World 1, there are clear income disparities by both gender and ethnicity, 
# with males and Ethnic Group 2 earning significantly more. 
# In contrast, World 2 demonstrates a highly equal income distribution 
# across all ethnic groups and genders, with both genders and all ethnic
#  groups earning around $60,000 supports both hypothesis. This consistent income equality in 
# World 2 aligns more closely with the utopian ideal of fairness and 
# equality, ensuring equal economic opportunities regardless of gender 
# or ethnicity. World 1’s evident disparities suggest structural inequalities that hinder 
# its alignment with utopian principles. Thus, World 2's balanced economic distribution positions 
# it as a fairer and more equitable society, reflecting the core tenets of a utopian world.


# %%

# Pivot table for World 1 (by gender and industry, showing average income and education)
pivot_world1 = pd.pivot_table(world1, values=['income00', 'education'], 
                              index=['gender', 'industry'], 
                              aggfunc={'income00': 'mean', 'education': 'mean'}).reset_index()

# Pivot table for World 2 (by gender and industry, showing average income and education)
pivot_world2 = pd.pivot_table(world2, values=['income00', 'education'], 
                              index=['gender', 'industry'], 
                              aggfunc={'income00': 'mean', 'education': 'mean'}).reset_index()

print("world1 pivot table\n",pivot_world1) 
print("\n world2 pivot table\n",pivot_world2)



# 1. Income: In both worlds, men consistently earn slightly more than women across all industries. The income gap is larger in industries with higher average salaries (such as Professional/Business and Finance). 
# This suggests gender disparity in income distribution, indicating that neither world is perfectly utopian in terms of gender income equality. However, World 2 shows a slightly smaller income gap overall compared to World 1.
# 2. Education: Education levels are quite similar across both worlds and are consistent across genders and industries, with an average of approximately 15 years of education. There doesn't appear to be any significant disparity in education access or outcomes, which is a positive indication for both worlds in terms of utopian characteristics.
# Both worlds show equality in terms of access to education, which is a positive characteristic in line with utopian ideals.
# Based on income and gender data, World 2 demonstrates slightly more utopian characteristics than World 1, as the income gap between genders is smaller. However, the persistent gender income gap in both worlds indicates that neither is a perfect utopia in this regard.

# 


# %%
# Gender representation in industries
plt.figure(figsize=(14, 6))

# World 1: Gender in Industries
plt.subplot(1, 2, 1)
sns.countplot(x='industry', hue='gender', data=world1)
plt.title('World 1: Gender Representation Across Industries')
plt.xticks(rotation=45)

# World 2: Gender in Industries
plt.subplot(1, 2, 2)
sns.countplot(x='industry', hue='gender', data=world2)
plt.title('World 2: Gender Representation Across Industries')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# World 1 shows some industries dominated by one gender (e.g., finance dominated by males, education by females), suggesting a lack of gender balance.
# World 2 demonstrates better gender representation across industries, with a more equal distribution of males and females.

# %%

# Plotting income distribution by ethnicity for World 1 and World 2
plt.figure(figsize=(14, 6))

# World 1: Income by Ethnicity
plt.subplot(1, 2, 1)
sns.boxplot(x='ethnic', y='income00', data=world1)
plt.title('World 1: Income Distribution by Ethnicity')

# World 2: Income by Ethnicity
plt.subplot(1, 2, 2)
sns.boxplot(x='ethnic', y='income00', data=world2)
plt.title('World 2: Income Distribution by Ethnicity')

plt.tight_layout()
plt.show()

# World 1 shows greater income disparity among ethnic groups, suggesting some level of marginalization.
# World 2 displays a more balanced income distribution across ethnic groups, indicating better ethnic equality, which aligns with utopian ideals.

# %%
# Plotting income by education level for both worlds
plt.figure(figsize=(14, 6))

# World 1: Income by Education
plt.subplot(1, 2, 1)
sns.boxplot(x='education', y='income00', data=world1)
plt.title('World 1: Income Distribution by Education')

# World 2: Income by Education
plt.subplot(1, 2, 2)
sns.boxplot(x='education', y='income00', data=world2)
plt.title('World 2: Income Distribution by Education')

plt.tight_layout()
plt.show()
# World 1 shows greater income inequality as education levels increase, with highly educated individuals earning significantly more.
# World 2 demonstrates more balanced income distribution across all education levels, indicating that education does not create massive income disparities, aligning with utopian ideals.

# %%
# Calculating mean and median income by gender and industry for both worlds
mean_income_world1 = world1.groupby(['gender', 'industry'])['income00'].agg(['mean', 'median']).reset_index()
mean_income_world2 = world2.groupby(['gender', 'industry'])['income00'].agg(['mean', 'median']).reset_index()

# Merging the data to compare mean and median income for both worlds
mean_income_world1, mean_income_world2

# World 1 shows a significant gender pay gap in several industries, with men earning more in both mean and median income across most industries.
# World 2 shows a smaller gender pay gap, particularly in industries like education and health, which are closer to gender parity, indicating it is closer to a utopian society where gender equality is prioritized.

# %%

# Plotting ethnic representation across industries for both worlds
plt.figure(figsize=(14, 6))

# World 1: Ethnic Representation in Industries
plt.subplot(1, 2, 1)
sns.countplot(x='industry', hue='ethnic', data=world1)
plt.title('World 1: Ethnic Representation Across Industries')

# World 2: Ethnic Representation in Industries
plt.subplot(1, 2, 2)
sns.countplot(x='industry', hue='ethnic', data=world2)
plt.title('World 2: Ethnic Representation Across Industries')

plt.tight_layout()
plt.show()

# World 1 shows that certain ethnic groups dominate higher-paying industries (e.g., finance, business), indicating ethnic inequality in terms of access to well-paying jobs.
# World 2 has more balanced ethnic representation across all industries, showing no clear concentration of one group in any industry, which suggests greater social fairness.

# %%
# https://en.wikipedia.org/wiki/Gini_coefficient   refrence for gini coeficient

# Calculating Gini coefficient for both worlds
def gini_coefficient(income):
    sorted_income = np.sort(income)
    n = len(income)
    cumulative_income = np.cumsum(sorted_income) / sorted_income.sum()
    index = np.arange(1, n+1)
    gini_index = (np.sum((2 * index - n - 1) * sorted_income)) / (n * sorted_income.sum())
    return gini_index

gini_world1 = gini_coefficient(world1['income00'])
gini_world2 = gini_coefficient(world2['income00'])

gini_world1, gini_world2
# The Gini coefficient measures income inequality, with 0 representing perfect equality (everyone earns the same income) and 1 representing maximum inequality (one person earns everything, while others earn nothing).
# The Gini coefficients for both World 1 and World 2 are very close to each other, at around 0.254–0.255, indicating that both worlds exhibit similar levels of income inequality.
# This similarity suggests that in terms of income equality, neither world is clearly more utopian than the other based solely on the Gini coefficient and noe of them are utopian
# But by considering previous analysis we can say that world2 is close to utopia.



















# %%