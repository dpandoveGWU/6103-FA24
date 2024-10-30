#Design a line graph to track the monthly growth rate of new subscribers to an online service, using 'blue' to represent the trend.

#Matplotlib Color Mapping: color = 'blue'Matplotlib Figure Size: figsize=(10, 6)Plotly Color Mapping: line_color = 'blue'
# %%
import matplotlib.pyplot as plt

# %%
months = ['January', 'February', 'March', 'April', 'May', 'June','July','August','Sept','Oct','Nov','Dec']
subscribers = [100, 150, 200, 250, 300, 400,150,320,170,500,222,345]

# %%
plt.figure(figsize=(10, 6))
plt.plot(months, subscribers, color='blue', marker='o')


plt.xlabel('Month')
plt.ylabel('New Subscribers')
plt.title('Monthly Growth Rate of New Subscribers')
plt.show()

# %%
