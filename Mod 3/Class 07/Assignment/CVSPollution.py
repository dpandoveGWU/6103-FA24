#Deploy a scatter plot to analyze the relationship between traffic congestion and air pollution in major cities, using 'gray' for low pollution and 'darkred' for high pollution.

#Matplotlib Color Mapping: colors = ['gray' if x < 80 else 'darkred' for x in df['air_pollution']]Matplotlib Figure Size: figsize=(10, 6)Plotly Color Mapping: color_discrete_map = {'Low Pollution': 'gray', 'High Pollution': 'darkred'}
# Cleaning up the dataframe by setting proper column names and removing the first row
# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
# importing the data set
df = pd.read_excel('cvp.xlsx')
print(df.head())

# %%
df.columns = ['City', 'traffic_congestion', 'air_pollution']
df = df.drop(0).reset_index(drop=True)

# Convert the columns to numeric, except for 'City'
df['traffic_congestion'] = pd.to_numeric(df['traffic_congestion'])
df['air_pollution'] = pd.to_numeric(df['air_pollution'])


#colors based on pollution level
colors = ['gray' if x < 80 else 'darkred' for x in df['air_pollution']]

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['traffic_congestion'], df['air_pollution'], c=colors)

#labels and title
plt.xlabel('Traffic Congestion (%)')
plt.ylabel('Air Pollution Level')
plt.title('Traffic Congestion vs Air Pollution in Major Cities')

# Show the plot
plt.show()

# %%
