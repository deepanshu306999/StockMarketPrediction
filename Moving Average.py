import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("TCS.csv")
print(df.head())
df_close = pd.DataFrame(df.Close)
df_close['MA_9'] = df_close.Close.rolling(9).mean()
df_close['MA_21'] = df_close.Close.rolling(21).mean()
plt.figure(figsize=(15,10))
plt.grid(True)
plt.plot(df_close['Close'],label='Close')
plt.plot(df_close['MA_9'],label='MA_9')
plt.plot(df_close['MA_21'],label='MA_21')
plt.legend(loc=2)
df_close['MA_9'].head(12)
df_close["change"] = np.log(df_close["Close"]/df_close["Close"].shift())
plt.plot(df_close.change)
df_close['Volatility'] = df_close.change.rolling(21).std().shift()
df_close['Volatility'].plot()
df_close['actual_change'] = df_close['Close'] - df_close['Close'].shift(1)
df_close['exp_change'] = df_close['Close'].shift(1) * df_close['Volatility']
df_close = df_close.iloc[1:]
df_close['Magnitude'] = df_close['actual_change'] / df_close['exp_change']
plt.hist(df_close['Magnitude'], bins=50)
df_close['abs_magni'] = np.abs(df_close['Magnitude'])
plt.scatter(df_close['actual_change'], df_close['abs_magni'])
plt.show()
