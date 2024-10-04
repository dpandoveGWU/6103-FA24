# %%
# Use Yahoo finance API
# %pip install yfinance
import yfinance as yf
import pandas as pd


#%%
# https://github.com/ranaroussi/yfinance 

msft = yf.Ticker("MSFT")
info = msft.info
info_df = pd.DataFrame(info)
info_df.head()

#%%
msft.info  # get stock info
msft.actions    # show actions (dividends, splits)
msft.dividends    # show dividends
msft.splits    # show splits

#%%
msft.financials    # show financials
# msft.quarterly_financials
# msft.major_holders    # show major holders
# msft.institutional_holders    # show institutional holders
# msft.balance_sheet    # show balance sheet
# msft.quarterly_balance_sheet

# %%
hist = msft.history(period="max")    # get historical market data
lastday = msft.history(period = '1d')   
lastclose = lastday['Close'][0]
print(f'last close = {lastclose}')


# %% 
import seaborn as sns
import matplotlib.pyplot as plt
sns.lineplot(x = 'Date', y = 'Close', data = hist)
plt.xticks(rotation = 45)
plt.show()
# %%
