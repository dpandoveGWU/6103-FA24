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

# From Box Plot1 an Box Plot2 we can see that 
# World 1: There's a noticeable income disparity among ethnic groups, especially with Ethnic Group 2 having significantly higher median incomes.
# World 2: All ethnic groups have similar median incomes, indicating a more uniform income distribution among different ethnicities.
# World 1 shows significant income disparities among ethnic groups, with Ethnic Group 2 earning considerably more than Groups 0 and 1. This suggests structural inequalities.
# World 2 displays more balanced income distribution across all ethnic groups, which aligns with the principles of fairness and equality, key aspects of a utopian society.
                              # Hypothesis 1:
# World 1 (left plot) shows a significant variation in income across different ethnic groups. Ethnic Group 2, in particular, has a much wider income range compared to the other two groups. This indicates notable disparities in income distribution across ethnicities, with Ethnic Group 2 enjoying higher incomes overall and greater variability.
# World 2 (right plot), on the other hand, exhibits a more balanced income distribution across all ethnic groups. The income range is narrower and more consistent, suggesting less income inequality among ethnic groups.
# Therefore, Hypothesis 1 is supported: The two worlds show distinct differences in income distribution by ethnicity, with World 2 being more equal and balanced.
                             #  Hypothesis 2:
# The more equal income distribution across ethnic groups in World 2 aligns with the ideals of a utopia, where income equality is more prevalent regardless of ethnic background. This suggests that World 2 fosters more social and economic fairness.
# In contrast, World 1 has greater income disparities, particularly with Ethnic Group 2 experiencing significantly higher incomes. This reflects a society with more inequality, far from the utopian ideal of fairness.
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

# Both World 1 and World 2 show that individuals with higher education levels (15+ years) have a broad range of incomes, 
# indicating diverse career outcomes. There’s a common income level around $60,000 for many individuals 
# regardless of their education level in both worlds. 
# Education appears to have a minimal impact on income, 
# with a significant variability seen at higher education levels. 
# Outliers exist in lower education levels, where some individuals 
# still achieve high incomes, though they are few. 
# Overall, education is not a sole determinant of income in 
# either world and not enough to identify or talk about utopia  .
# %%
# 4. Marital Status vs. Income

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


# %%
# 5. Age vs. Income Analysis

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

# The income distribution by marital status shows consistent median incomes
#  around $60,000 for all groups in both World 1 and World 2.
#  This suggests that marital status does not significantly 
# impact income. Since utopia emphasizes fairness and equality, 
# this consistency implies that neither world allows marital status 
# to influence economic status unduly. Therefore, based on marital status 
# alone,neither world can be determined to be more utopian.


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
#  groups earning around $60,000. This consistent income equality in 
# World 2 aligns more closely with the utopian ideal of fairness and 
# equality, ensuring equal economic opportunities regardless of gender 
# or ethnicity. World 1’s evident disparities suggest structural inequalities that hinder 
# its alignment with utopian principles. Thus, World 2's balanced economic distribution positions 
# it as a fairer and more equitable society, reflecting the core tenets of a utopian world.



# %%
# Pivot Table for Average Income by Industry
pivot_world1 = world1.pivot_table(values='income00', index='industry', aggfunc='mean')
pivot_world2 = world2.pivot_table(values='income00', index='industry', aggfunc='mean')
print("World 1: Average Income by Industry")
print(pivot_world1)
print("\nWorld 2: Average Income by Industry")
print(pivot_world2)

# Both world1 and world2 have similar trend(have higher average as industry number increases)
# which shows consistent economic structure
# World 2’s slight reduction in disparity and its generally higher average incomes align better with 
# the utopian ideal of fairness and economic balance.



# %%
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(world1['income00'], world2['income00'])

print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# The t-test result shows no significant difference in overall income 
# distributions between World 1 and World 2, 
# with a t-statistic of 0.0289 and a p-value of 0.977. 
# The high p-value indicates that any observed difference in 
# means is likely due to random chance, not an actual difference
#  in income levels. This suggests that both worlds have similar 
# overall income levels. However, detailed analyses reveal that 
# World 2 has better gender and ethnic income equality. 
# Therefore, despite similar overall income levels, 
# World 2 is closer to the utopian ideal of fairness and equality.


# %%





# %%
# Pivot table for mean income by marital status and industry in world1
world1_pivot = world1.pivot_table(values="income00", index="marital", columns="industry", aggfunc=np.mean)
print("Mean Income by Marital Status and Industry in World 1:\n", world1_pivot)

# Pivot table for mean income by marital status and industry in world2
world2_pivot = world2.pivot_table(values="income00", index="marital", columns="industry", aggfunc=np.mean)
print("Mean Income by Marital Status and Industry in World 2:\n", world2_pivot)

# %%
plt.figure(figsize=(10, 5))
sns.histplot(world1["age00"], kde=True, label="World 1", color="blue")
sns.histplot(world2["age00"], kde=True, label="World 2", color="red")
plt.title("Age Distribution in World 1 and World 2")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 5))
sns.histplot(world1["education"], kde=True, label="World 1", color="blue")
sns.histplot(world2["education"], kde=True, label="World 2", color="red")
plt.title("Education Level Distribution in World 1 and World 2")
plt.xlabel("Years of Education")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
marital_counts1 = world1["marital"].value_counts(normalize=True)
marital_counts2 = world2["marital"].value_counts(normalize=True)

plt.figure(figsize=(12, 6))
marital_counts1.plot(kind="bar", alpha=0.5, label="World 1", color="blue")
marital_counts2.plot(kind="bar", alpha=0.5, label="World 2", color="red")
plt.title("Marital Status Proportions in World 1 and World 2")
plt.xlabel("Marital Status")
plt.ylabel("Proportion")
plt.legend()
plt.show()

# %%
gender_counts1 = world1["gender"].value_counts(normalize=True)
gender_counts2 = world2["gender"].value_counts(normalize=True)

plt.figure(figsize=(12, 6))
gender_counts1.plot(kind="bar", alpha=0.5, label="World 1", color="blue")
gender_counts2.plot(kind="bar", alpha=0.5, label="World 2", color="red")
plt.title("Gender Distribution in World 1 and World 2")
plt.xlabel("Gender")
plt.ylabel("Proportion")
plt.legend()
plt.show()

# %%
ethnic_counts1 = world1["ethnic"].value_counts(normalize=True)
ethnic_counts2 = world2["ethnic"].value_counts(normalize=True)

plt.figure(figsize=(12, 6))
ethnic_counts1.plot(kind="bar", alpha=0.5, label="World 1", color="blue")
ethnic_counts2.plot(kind="bar", alpha=0.5, label="World 2", color="red")
plt.title("Ethnic Distribution in World 1 and World 2")
plt.xlabel("Ethnic Group")
plt.ylabel("Proportion")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(14, 7))
sns.boxplot(x="industry", y="income00", data=world1)
plt.title("Income Distribution by Industry in World 1")
plt.xlabel("Industry")
plt.ylabel("Income")
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(x="industry", y="income00", data=world2)
plt.title("Income Distribution by Industry in World 2")
plt.xlabel("Industry")
plt.ylabel("Income")
plt.show()

# %%
plt.figure(figsize=(14, 7))
sns.boxplot(x="gender", y="income00", data=world1)
plt.title("Income Disparity by Gender in World 1")
plt.xlabel("Gender")
plt.ylabel("Income")
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(x="gender", y="income00", data=world2)
plt.title("Income Disparity by Gender in World 2")
plt.xlabel("Gender")
plt.ylabel("Income")
plt.show()

# %%
plt.figure(figsize=(14, 7))
sns.boxplot(x="marital", y="income00", data=world1)
plt.title("Income Disparity by Marital Status in World 1")
plt.xlabel("Marital Status")
plt.ylabel("Income")
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(x="marital", y="income00", data=world2)
plt.title("Income Disparity by Marital Status in World 2")
plt.xlabel("Marital Status")
plt.ylabel("Income")
plt.show()

# %%
plt.figure(figsize=(14, 7))
sns.heatmap(world1.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for World 1")
plt.show()

plt.figure(figsize=(14, 7))
sns.heatmap(world2.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for World 2")
plt.show()

# %%
from scipy.stats import ttest_ind

# T-test for overall income
t_stat, p_val = ttest_ind(world1["income00"], world2["income00"])
print("T-test between World 1 and World 2 incomes: t-stat =", t_stat, ", p-value =", p_val)

# T-test for income by gender
t_stat1, p_val1 = ttest_ind(world1[world1["gender"] == 0]["income00"], world1[world1["gender"] == 1]["income00"])
t_stat2, p_val2 = ttest_ind(world2[world2["gender"] == 0]["income00"], world2[world2["gender"] == 1]["income00"])
print("T-test for gender income disparity in World 1: t-stat =", t_stat1, ", p-value =", p_val1)
print("T-test for gender income disparity in World 2: t-stat =", t_stat2, ", p-value =", p_val2)

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



# %%
def plot_income_distribution(data1, data2, column, title):
    plt.figure(figsize=(14, 7))
    sns.histplot(data1[column], kde=True, label="World 1", color="blue")
    sns.histplot(data2[column], kde=True, label="World 2", color="red")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_box(data, x, y, hue, title):
    plt.figure(figsize=(14, 7))
    sns.boxplot(x=x, y=y, hue=hue, data=data)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=hue)
    plt.show()

# Plot income distribution by industry and gender for both worlds
plot_box(world1, "industry", "income00", "gender", "Income Distribution by Industry and Gender in World 1")
plot_box(world2, "industry", "income00", "gender", "Income Distribution by Industry and Gender in World 2")

# Compare overall income distributions
plot_income_distribution(world1, world2, "income00", "Overall Income Distribution Comparison")


# %%
from scipy.stats import ttest_ind

def conduct_t_test(group1, group2, description):
    t_stat, p_val = ttest_ind(group1, group2)
    print(f"T-test {description}: t-stat =", t_stat, ", p-value =", p_val)

# T-test for overall income
conduct_t_test(world1["income00"], world2["income00"], "between World 1 and World 2 incomes")

# T-test for income by gender
conduct_t_test(world1[world1["gender"] == 0]["income00"], world1[world1["gender"] == 1]["income00"], "for gender income disparity in World 1")
conduct_t_test(world2[world2["gender"] == 0]["income00"], world2[world2["gender"] == 1]["income00"], "for gender income disparity in World 2")

# %%