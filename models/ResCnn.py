import torch
import torch.nn as nn


class ResCnn(nn.Module):
    def __init__(self,
                 config: dict,
                 ):
        super(ResCnn, self).__init__()

    def forward(self, x):
        print(x)
        mean_0 = torch.mean(x[:, :, :], dim=1, keepdim=True)
        print(mean_0)
        x = (x-mean_0)
        print(x)
        mean_1 = torch.mean(x[:, 20:30, :], dim=1, keepdim=True)
        print(mean_1)
        x = (x[:, 20:30, :] - mean_1)
        mean_2 = torch.mean(x[:, 5:10, :], dim=1, keepdim=True)
        print(mean_2)
        x = (x[:, 5:10, :] - mean_2)

        print(x)
        return x


import pandas as pd

df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
x_in = torch.from_numpy(df.iloc[-3 * 10:, :-1].to_numpy())
x_in = x_in.reshape(1, *x_in.size())
# x_in = torch.tensor(x_in, dtype=torch.float32)
print(x_in.shape)
model = ResCnn({})
model(x_in)
