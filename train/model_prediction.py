import torch

from post_process import PrePostProcess
from models.RNN import LSTMModel
from models.CausalRnn import CausalRnn

import pickle
import numpy as np
import pandas as pd

with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)

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


df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
print(df)

window = config['number_days'] * config['tick_per_day']
input_to_predict = df.iloc[-1 * window:].copy()
proc = PrePostProcess(input_to_predict)
nn_input_numpy = proc.obs()  # add dummy dim for batch
nn_input_torch = torch.from_numpy(np.expand_dims(nn_input_numpy, 0)).type(torch.float32)
print(nn_input_torch.shape)
x = model(nn_input_torch)
print(x.detach().numpy())
# nn_output_numpy = MODEL ( nn_input_torch).numpy()

nn_output_numpy = x.detach().numpy()  # proc.target()
df_predict = proc.prediction_to_ohlc_df(nn_output_numpy)
df_obs = proc.ohlc_obs
df_target = proc.ohlc_target
print(df_obs)
#print(df_target)
print(df_predict)
