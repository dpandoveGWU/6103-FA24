# %%
# EAP Visualization
import matplotlib.pyplot as plt
# %%
# Data for Netflix subscriber growth (in millions)
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
forecasted_growth = [1.1, 1.15, 1.0, 1.2]  # forecasted subscriber growth in millions
actual_growth = [1.1, 0.88, 1.05, 1.3]     # actual subscriber growth in millions

# Data for Netflix stock price reaction
days = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']
stock_prices = [500, 425, 440, 445, 490]  # stock price after report (simulated)

# Creating the figure for subscriber growth comparison
plt.figure(figsize=(10, 6))

# Plotting forecasted vs actual growth
plt.subplot(1, 2, 1)
plt.plot(quarters, forecasted_growth, label="Forecasted Growth", marker='o', color='green')
plt.plot(quarters, actual_growth, label="Actual Growth", marker='o', color='red')
plt.title('Netflix Subscriber Growth (Forecasted vs Actual)')
plt.xlabel('Quarter')
plt.ylabel('Subscribers (in millions)')
plt.legend()

# Creating the figure for stock price reaction
plt.subplot(1, 2, 2)
plt.plot(days, stock_prices, label="Stock Price", marker='o', color='blue')
plt.title('Netflix Stock Price Reaction')
plt.xlabel('Days After Report')
plt.ylabel('Stock Price ($)')
plt.legend()

# Adjusting layout for better view
plt.tight_layout()

# Show the plots
plt.show()

# %%
