import pandas as pd
import torch

df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
# print(df)
x = df[['high', 'low', 'close']].iloc[-1, :].to_list()
# print(x)
x = torch.Tensor(x)
print(x)
print(y)
print(x - y)
ll = torch.nn.MSELoss()
loss = torch.sqrt(ll(x, y))
print(loss)
