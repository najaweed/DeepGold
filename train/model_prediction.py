import torch

from post_process import PrePostProcess
from models.RNN import LSTMModel
from models.CausalRnn import CausalRnn
from preprocess import NyDiffNormalizer
import pickle
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# from mt5 import ForexMarket

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()


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
xau_df = get_last_day('XAUUSD', window)
# print(xau_df)
t_df = session_transform(sessions=s_sessions, x_df=xau_df)
# print(t_df)
xx_df = group_data('1D', t_df)
# print(xx_df)

# breakpoint()
model = CausalRnn(config)
checkpoint = torch.load("best_model.ckpt")
model_weights = checkpoint["state_dict"]
# update keys by dropping `nn_model.`
for key in list(model_weights):
    model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
model.load_state_dict(model_weights)
model.eval()
print('model loaded with learned parameters, ready for predict')
#
#  a function for model prediction with dataframe input and plot result of model and dataframe


# df = pd.read_csv('gold.csv', )  # , parse_dates=True)
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)
# print(df)

window = config['number_days'] * config['tick_per_day']
input_to_predict = xx_df  # df.iloc[-1 * window:].copy()
print(input_to_predict)
# nydiff = NyDiffNormalizer(ohlc)
proc = NyDiffNormalizer(input_to_predict.iloc[:-1, :])
nn_input_numpy = proc.obs()  # add dummy dim for batch
nn_input_torch = torch.from_numpy(np.expand_dims(nn_input_numpy, 0)).type(torch.float32)
print(nn_input_torch.shape)
x = model(nn_input_torch)
print(x.detach().numpy())
# nn_output_numpy = MODEL ( nn_input_torch).numpy()

nn_output_numpy = x.detach().numpy()  # proc.target()

df_predict = proc.prediction_to_ohlc_df(nn_output_numpy)

print(proc.df.iloc[:-1, :])
print(proc.df.iloc[-1, :])
print(df_predict)
