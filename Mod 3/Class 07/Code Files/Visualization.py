#%% 
#Data Visuaization using Matplotlib
#pip install matplotlib

#Below is a brief matplotlib API primer

# %%
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
pd.options.display.max_columns = 20
np.random.seed(12345)
import matplotlib.pyplot as plt
import matplotlib
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# %%
import matplotlib.pyplot as plt

# %%
#Simple plot
data = np.arange(10)
data
plt.plot(data)

# %%
##Plots in matplotlib reside within a Figure object. You can create a new figure with plt.figure:
fig = plt.figure()

# %%
#You have to create one or more subplots using add_subplot:
ax1 = fig.add_subplot(2, 2, 1)
#This means that the figure should be 2 × 2 (so up to four plots in total), and we’re selecting the first of four subplots (numbered from 1)

# %%
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
#%%
#Run this cell
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

# %%
#A line plot
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

ax3.plot(np.random.standard_normal(50).cumsum(), color="black",
         linestyle="dashed")

# %%
#Histogram
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax1.hist(np.random.standard_normal(100), bins=20, color="black", alpha=0.3)
#semicolan in the end suppresses the output
#Scatterplot
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30))
ax3.plot(np.random.standard_normal(50).cumsum(), color="black",
         linestyle="dashed");


# %%
plt.close("all")

# %%
fig, axes = plt.subplots(2, 3)
axes
#The axes array can then be indexed like a two-dimensional array; for example, axes[0, 1] refers to the subplot in the top row at the center.

# %%
#wspace and hspace control the percent of the figure width and figure height, respectively, to use as spacing between subplots.
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.standard_normal(500), bins=50,
                        color="black", alpha=0.5)
fig.subplots_adjust(wspace=0, hspace=0)

#colors, markers, line styles
#Line plots can additionally have markers to highlight the actual data points. Since matplotlib's plot function creates a continuous line plot, interpolating between points, it can occasionally be unclear where the points lie. The marker can be supplied as an additional styling option

# %%
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.random.standard_normal(30).cumsum(), color="black",
        linestyle="dashed", marker="o");
#Specify circle market

# %%
plt.close("all")

# %%
fig = plt.figure()
ax = fig.add_subplot()
data = np.random.standard_normal(30).cumsum()
ax.plot(data, color="black", linestyle="dashed", label="Default");
ax.plot(data, color="black", linestyle="dashed",
        drawstyle="steps-post", label="steps-post");
ax.legend()

# %%
#Ticks, Labels, and Legends
#Start by creating a simple plot
fig, ax = plt.subplots()
ax.plot(np.random.standard_normal(1000).cumsum());

# %%
#To change the x-axis ticks, it’s easiest to use set_xticks and set_xticklabels
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(["one", "two", "three", "four", "five"],
                            rotation=30, fontsize=8)
#The rotation option sets the x tick labels at a 30-degree rotation. Lastly, set_xlabel gives a name to the x-axis, and set_title is the subplot title
# %%
ax.set_xlabel("Stages")
ax.set_title("My first matplotlib plot")

#Bring these all together to edit the first plot
# %%
#Adding Legends
fig, ax = plt.subplots()
ax.plot(np.random.randn(1000).cumsum(), color="black", label="one");
ax.plot(np.random.randn(1000).cumsum(), color="black", linestyle="dashed",
        label="two");
ax.plot(np.random.randn(1000).cumsum(), color="black", linestyle="dotted",
        label="three");
ax.legend()


# %%
#Annotations and drawing on a subplot

from datetime import datetime

fig, ax = plt.subplots()

data = pd.read_csv("spx.csv", index_col=0, parse_dates=True)
spx = data["SPX"]

spx.plot(ax=ax, color="black")

crisis_data = [
    (datetime(2007, 10, 11), "Peak of bull market"),
    (datetime(2008, 3, 12), "Bear Stearns Fails"),
    (datetime(2008, 9, 15), "Lehman Bankruptcy")
]

for date, label in crisis_data:
    ax.annotate(label, xy=(date, spx.asof(date) + 75),
                xytext=(date, spx.asof(date) + 225),
                arrowprops=dict(facecolor="black", headwidth=4, width=2,
                                headlength=4),
                horizontalalignment="left", verticalalignment="top")

# Zoom in on 2007-2010
ax.set_xlim(["1/1/2007", "1/1/2011"])
ax.set_ylim([600, 1800])

ax.set_title("Important dates in the 2008–2009 financial crisis")

#The ax.annotate method can draw labels at the indicated x and y coordinates. 
# We use the set_xlim and set_ylim methods to manually set the start and end boundaries for the plot rather than using matplotlib's default. 
# Lastly, ax.set_title adds a main title to the plot.


# %%
#To add a shape to a plot, you create the patch object and add it to a subplot ax by passing the patch to ax.add_patch
fig, ax = plt.subplots(figsize=(12, 6))
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
                   color="green", alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
#%%
#Save fig 
fig.savefig("figpath.svg")
fig.savefig("figpdf.pdf")

# %%
plt.close("all")

# %%
#Plotting with Pandas and Seaborn
s = pd.Series(np.random.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
#he Series object’s index is passed to matplotlib for plotting on the x-axis, though you can disable this by passing use_index=False. 
# The x-axis ticks and limits can be adjusted with the xticks and xlim options, and the y-axis respectively with yticks and ylim

# %%
df = pd.DataFrame(np.random.standard_normal((10, 4)).cumsum(0),
                  columns=["A", "B", "C", "D"],
                  index=np.arange(0, 100, 10))
plt.style.use('grayscale')
df.plot()
#The plot.bar() and plot.barh() make vertical and horizontal bar plots, respectively. In this case, the Series or DataFrame index will be used as the x (bar) or y (barh) ticks

# %%
#Bar Plot
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.uniform(size=16), index=list("abcdefghijklmnop"))
data.plot.bar(ax=axes[0], color="black", alpha=0.7)
data.plot.barh(ax=axes[1], color="black", alpha=0.7)

# %%
np.random.seed(12348)

# %%
df = pd.DataFrame(np.random.uniform(size=(6, 4)),
                  index=["one", "two", "three", "four", "five", "six"],
                  columns=pd.Index(["A", "B", "C", "D"], name="Genus"))
df
df.plot.bar()

# %%
#plt.figure()

# %%
df.plot.barh(stacked=True, alpha=0.5)

# %%
plt.close("all")

# %%
tips = pd.read_csv("tips.csv")
tips.head()
party_counts = pd.crosstab(tips["day"], tips["size"])
party_counts = party_counts.reindex(index=["Thur", "Fri", "Sat", "Sun"])
party_counts

# %%
party_counts = party_counts.loc[:, 2:5]
party_counts

# %%
#Fraction of parties by size within each day
# party_pcts, is a DataFrame where each value represents the percentage (or proportion) of the total for that particular row, calculated as:
party_pcts = party_counts.div(party_counts.sum(axis="columns"),
                              axis="index")
party_pcts
party_pcts.plot.bar(stacked=True)

# %%
plt.close("all")

# %%
#seaborn
import seaborn as sns

tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
print(tips.head())
sns.barplot(x="tip_pct", y="day", data=tips, orient="h")
#The black lines drawn on the bars represent the 95% confidence interval (this can be configured through optional arguments).

# %%
plt.close("all")

# %%
#seaborn.barplot has a hue option that enables us to split by an additional categorical value
sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")

# %%
plt.close("all")

# %%
sns.set_style("whitegrid")

# %%
plt.figure()

# %%
#Histogram
tips["tip_pct"].plot.hist(bins=50)

# %%
#plt.figure()

# %%
#Density plot
#pip install scipy
tips["tip_pct"].plot.density()

# %%
#plt.figure()

# %%
#seaborn makes histograms and density plots even easier through its histplot method, which can plot both a histogram and a continuous density estimate simultaneously.
comp1 = np.random.standard_normal(200)
comp2 = 10 + 2 * np.random.standard_normal(200)
values = pd.Series(np.concatenate([comp1, comp2]))

sns.histplot(values, bins=100, color="black")

# %%
#We can then use seaborn's regplot method, which makes a scatter plot and fits a linear regression line
#Point plots or scatter plots can be a useful way of examining the relationship between two one-dimensional data series. 
# For example, here we load the macrodata dataset from the statsmodels project, select a few variables, then compute log differences
macro = pd.read_csv("macrodata.csv")
data = macro[["cpi", "m1", "tbilrate", "unemp"]]
trans_data = np.log(data).diff().dropna()
trans_data.tail()

# %%
#plt.figure()

# %%
#We can then use seaborn's regplot method, which makes a scatter plot and fits a linear regression line
ax = sns.regplot(x="m1", y="unemp", data=trans_data)
ax.set_title("Changes in log(m1) versus log(unemp)")

# %%
#In exploratory data analysis, it’s helpful to be able to look at all the scatter plots among a group of variables;
# this is known as a pairs plot or scatter plot matrix. 
# Making such a plot from scratch is a bit of work,
# so seaborn has a convenient pairplot function that supports placing 
# histograms or density estimates of each variable along the diagonal
sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})

# %%
#Facet Grids and Categorical Data
sns.catplot(x="day", y="tip_pct", hue="time", col="smoker",
            kind="bar", data=tips[tips.tip_pct < 1])

# %%
#Instead of grouping by "time" by different bar colors within a facet, we can also expand the facet grid by adding one row per time value
sns.catplot(x="day", y="tip_pct", row="time",
            col="smoker",
            kind="bar", data=tips[tips.tip_pct < 1])

# %%
sns.catplot(x="tip_pct", y="day", kind="box",
            data=tips[tips.tip_pct < 0.5])

# %%


# %%
pd.options.display.max_rows = PREVIOUS_MAX_ROWS


