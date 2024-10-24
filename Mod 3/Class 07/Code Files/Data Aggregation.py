#Data Aggregation and Group Operations

# %%
import numpy as np
import pandas as pd
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(10, 6))
np.set_printoptions(precision=4, suppress=True)

# %%
import numpy as np
import pandas as pd

# %%
df = pd.DataFrame({"key1" : ["a", "a", None, "b", "b", "a", None],
                   "key2" : pd.Series([1, 2, 1, 2, 1, None, 1],
                                      dtype="Int64"),
                   "data1" : np.random.standard_normal(7),
                   "data2" : np.random.standard_normal(7)})
print(df)

# %%
grouped = df["data1"].groupby(df["key1"])

# %%
grouped.mean()

# %%
means = df["data1"].groupby([df["key1"], df["key2"]]).mean()
means

# %%
#Here we grouped the data using two keys, and the resulting Series now has a hierarchical index consisting of the unique pairs of keys observed:


means.unstack()
# %%
#In this example, the group keys are all Series, though they could be any arrays of the right length:
states = np.array(["OH", "CA", "CA", "OH", "OH", "CA", "OH"])
years = [2005, 2005, 2006, 2005, 2006, 2005, 2006]
df["data1"].groupby([states, years]).mean()

# %%
print(df.groupby("key1").mean())
print(df.groupby("key2").mean(numeric_only=True))
print(df.groupby(["key1", "key2"]).mean())
#notice that in the second case, it is necessary to pass numeric_only=True because the key1 column is not numeric and thus cannot be aggregated with mean().

# %%
#GroupBy method is size, which returns a Series containing group sizes:
df.groupby(["key1", "key2"]).size()


# %%

print(df.groupby("key1", dropna=False).size())
print(df.groupby(["key1", "key2"], dropna=False).size())

# %%
#A group function similar in spirit to size is count, which computes the number of nonnull values in each group:
df.groupby("key1").count()

# %%
#Iterating over Groups
for name, group in df.groupby("key1"):
    print(name)
    print(group)


# %%

for (k1, k2), group in df.groupby(["key1", "key2"]):
    print((k1, k2))
    print(group)


# %%
#computing a dictionary of the data pieces as a one-liner
pieces = {name: group for name, group in df.groupby("key1")}
#pieces
pieces["b"]

# %%
grouped = df.groupby({"key1": "key", "key2": "key",
                      "data1": "data", "data2": "data"}, axis="columns")
print(grouped)
# %%
for group_key, group_values in grouped:
    print(group_key)
    print(group_values)


# %%
#Selecting a Column or Subset of Columns
df.groupby(["key1", "key2"])[["data2"]].mean()

# %%
s_grouped = df.groupby(["key1", "key2"])["data2"]
s_grouped
s_grouped.mean()

# %%
#Grouping with Dictionaries and Series
people = pd.DataFrame(np.random.standard_normal((5, 5)),
                      columns=["a", "b", "c", "d", "e"],
                      index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
people.iloc[2:3, [1, 2]] = np.nan # Add a few NA values
people

# %%
mapping = {"a": "red", "b": "red", "c": "blue",
           "d": "blue", "e": "red", "f" : "orange"}

# %%
by_column = people.groupby(mapping, axis="columns")
by_column.sum()

# %%
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis="columns").count()

# %%
#Grouping with Functions
people.groupby(len).sum()

# %%

key_list = ["one", "one", "one", "two", "two"]
people.groupby([len, key_list]).min()

# %%
#Grouping by Index Levels
columns = pd.MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"],
                                    [1, 3, 5, 1, 3]],
                                    names=["cty", "tenor"])
hier_df = pd.DataFrame(np.random.standard_normal((4, 5)), columns=columns)
hier_df

# %%

hier_df.groupby(level="cty", axis="columns").count()

# %%
df
grouped = df.groupby("key1")
grouped["data1"].nsmallest(2)

# %%
def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped.agg(peak_to_peak)

# %%
grouped.describe()

# %%
tips = pd.read_csv("tips.csv")
tips.head()

# %%
tips["tip_pct"] = tips["tip"] / tips["total_bill"]
tips.head()

# %%
grouped = tips.groupby(["day", "smoker"])

# %%
grouped_pct = grouped["tip_pct"]
grouped_pct.agg("mean")

# %%
grouped_pct.agg(["mean", "std", peak_to_peak])

# %%
grouped_pct.agg([("average", "mean"), ("stdev", np.std)])

# %%
functions = ["count", "mean", "max"]
result = grouped[["tip_pct", "total_bill"]].agg(functions)
result

# %%
result["tip_pct"]

# %%
ftuples = [("Average", "mean"), ("Variance", np.var)]
grouped[["tip_pct", "total_bill"]].agg(ftuples)

# %%
grouped.agg({"tip" : np.max, "size" : "sum"})
grouped.agg({"tip_pct" : ["min", "max", "mean", "std"],
             "size" : "sum"})

# %%
grouped = tips.groupby(["day", "smoker"], as_index=False)
grouped.mean(numeric_only=True)

# %%
def top(df, n=5, column="tip_pct"):
    return df.sort_values(column, ascending=False)[:n]
top(tips, n=6)

# %%
tips.groupby("smoker").apply(top)

# %%
tips.groupby(["smoker", "day"]).apply(top, n=1, column="total_bill")

# %%
result = tips.groupby("smoker")["tip_pct"].describe()
result
result.unstack("smoker")

# %%
tips.groupby("smoker", group_keys=False).apply(top)




# %%
#Filling Missing Values with Group-Specific Values
s = pd.Series(np.random.standard_normal(6))
s[::2] = np.nan
s
s.fillna(s.mean())

# %%
states = ["Ohio", "New York", "Vermont", "Florida",
          "Oregon", "Nevada", "California", "Idaho"]
group_key = ["East", "East", "East", "East",
             "West", "West", "West", "West"]
data = pd.Series(np.random.standard_normal(8), index=states)
data

# %%
data[["Vermont", "Nevada", "Idaho"]] = np.nan
data
data.groupby(group_key).size()
data.groupby(group_key).count()
data.groupby(group_key).mean()

# %%
def fill_mean(group):
    return group.fillna(group.mean())

data.groupby(group_key).apply(fill_mean)

# %%
fill_values = {"East": 0.5, "West": -1}
def fill_func(group):
    return group.fillna(fill_values[group.name])

data.groupby(group_key).apply(fill_func)




# %%
#Example: Group wise linear regression
import statsmodels.api as sm
def regress(data, yvar=None, xvars=None):
    Y = data[yvar]
    X = data[xvars]
    X["intercept"] = 1.
    result = sm.OLS(Y, X).fit()
    return result.params

# %%
by_year.apply(regress, yvar="AAPL", xvars=["SPX"])

# %%
#Pivot Tables and Cross-Tabulation
print(tips.head())
tips.pivot_table(index=["day", "smoker"],
                 values=["size", "tip", "tip_pct", "total_bill"])

# %%
tips.pivot_table(index=["time", "day"], columns="smoker",
                 values=["tip_pct", "size"])

# %%
tips.pivot_table(index=["time", "day"], columns="smoker",
                 values=["tip_pct", "size"], margins=True)

# %%
tips.pivot_table(index=["time", "smoker"], columns="day",
                 values="tip_pct", aggfunc=len, margins=True)

# %%
tips.pivot_table(index=["time", "size", "smoker"], columns="day",
                 values="tip_pct", fill_value=0)

# %%
from io import StringIO
data = """Sample  Nationality  Handedness
1   USA  Right-handed
2   Japan    Left-handed
3   USA  Right-handed
4   Japan    Right-handed
5   Japan    Left-handed
6   Japan    Right-handed
7   USA  Right-handed
8   USA  Left-handed
9   Japan    Right-handed
10  USA  Right-handed"""
data = pd.read_table(StringIO(data), sep="\s+")

# %%
data

# %%
pd.crosstab(data["Nationality"], data["Handedness"], margins=True)

# %%
pd.crosstab([tips["time"], tips["day"]], tips["smoker"], margins=True)

# %%


# %%
pd.options.display.max_rows = PREVIOUS_MAX_ROWS


