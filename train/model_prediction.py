import torch

from models.ResCnn import ResidualStack
# from post_process import PrePostProcess
# from models.RNN import LSTMModel
# from models.CausalRnn import CausalRnn
# from models.AutoEncoder import Autoencoder

from pre_process import NyDiffNormalizer
import pickle
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

#from train.EncoderLSTM import EncoderLSTM

#from train.EncoderConv import EncoderConv

# from mt5 import ForexMarket




def get_last_day(symbol: str, shift_5m=400, s=0):
    ticks = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, s, shift_5m))
    ticks['time'] = pd.to_datetime(ticks['time'], unit='s')
    ticks = ticks.set_index('time')
    return ticks


def group_data(freq, x_df):
    df = x_df.copy()
    df['time'] = df.index
    df_list = []
    for group_name, df_group in df.groupby(pd.Grouper(freq=freq, key='time')):
        g_df = df_group.drop(columns=['time'])
        # week_or_month = 20 if self.freq[1] == 'M' else 5
        # if len(g_df) == 3:  # 5days*5weeks
        df_list.append(g_df)
    return pd.DataFrame(pd.concat(df_list, axis=0).sort_index())


def session_transform(sessions, x_df):
    t_df = []
    ohlc = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'tick_volume': 'sum'
    }

    for i, session in enumerate(sessions):
        # print(session[-1][:2])
        # h_seassion = session[-1][:2]
        df = x_df.copy().between_time(*session)
        df = df.resample('1D').apply(ohlc)
        df['time'] = df.index
        df['time'] = df['time'] + pd.Timedelta(hours=i)
        # print(df)
        df = df.set_index('time')
        t_df.append(df)

    return pd.DataFrame(pd.concat(t_df, axis=0).dropna().sort_index())


s_sessions = [('00:01', '7:00'), ('07:01', '13:00'), ('13:01', '23:59')]
# print(t_df.to_numpy())
# import numpy as np

# print(np.diff(t_df.to_numpy(), axis=0))

with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)
window = config['number_days'] * config['tick_per_day']
#['bottleneck_channels'] = 4
#config['hidden_channels'] = 16
# xau_df = get_last_day('XAUUSD', window)
# print(xau_df)
# t_df = session_transform(sessions=s_sessions, x_df=xau_df)
# print(t_df)
# xx_df = group_data('1D', t_df)
# print(xx_df)

# breakpoint()
# model = CausalRnn(config)
# model = Autoencoder(config)
model = ResidualStack(config)
checkpoint = torch.load("conv_params.ckpt")
model_weights = checkpoint["state_dict"]

# update keys by dropping `nn_model.`
for key in list(model_weights):
    model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights)
model.eval()
print('model loaded with learned parameters, ready for predict')
#
#  a function for model prediction with dataframe input and plot result of model and dataframe


df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
#print(df)
#df = df.iloc[:-1,:]
#print(df)

window = config['number_days'] * config['tick_per_day']
a = 10 # -a*config['tick_per_day']
input_to_predict = df.iloc[-1 * window-a*config['tick_per_day']:].copy()
# print(input_to_predict)
# nydiff = NyDiffNormalizer(ohlc)
proc = NyDiffNormalizer(input_to_predict.iloc[:, :])
nn_input_numpy = proc.obs()  # add dummy dim for batch
nn_input_torch = torch.from_numpy(np.expand_dims(nn_input_numpy, 0)).type(torch.float32)
# print(nn_input_torch.shape)
nn_input_torch = torch.permute(nn_input_torch, (0, 2, 1))
#print(nn_input_torch)
#print(nn_input_torch.shape)
#breakpoint()
x = model(nn_input_torch)
# encode_x = model.encoder(nn_input_torch)[0]
# print(proc.target())
x_loss = torch.nn.MSELoss()
x_1, x_2 = x[0], torch.from_numpy(proc.target())

nn_output_numpy = x.detach().numpy()  # proc.target()
# print(nn_output_numpy)
nn_output_numpy = nn_output_numpy[0].T
x_in = nn_input_torch.detach().numpy()[0].T

df_predict = proc.prediction_to_ohlc_df(nn_output_numpy)

print(proc.df.iloc[-3:-1, :])
print(proc.df.iloc[-1, :])
print(df_predict)
# import matplotlib.pyplot as plt
print(x_1, x_2)
print(torch.sqrt(x_loss(x_1, x_2)))
# plt.figure(1)
# plt.plot(nn_output_numpy)
# plt.plot(x_in, '.-', c='y')
# plt.figure(2)
# plt.imshow(x_encode)
# plt.show()
