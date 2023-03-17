import torch
import numpy as np
import pandas as pd


class PrePostProcess:
    def __init__(self,
                 df: pd.DataFrame
                 ):
        self.df = df
        # print(self.df)
        # print(np.diff(self.df.to_numpy(), axis=0))
        self.ts = np.diff(self.df.to_numpy(), axis=0)  # self.df  # np.diff(self.df.to_numpy(), axis=0)
        self.scale_normal = None
        self.min = None
        self.ohlc_obs = df.iloc[:-1, :].copy()
        self.ohlc_target = df.iloc[-1, :].copy()

    def obs(self):
        # ts_obs = self.ts.to_numpy()[:-1, :]
        ts_obs = self.ts[:-1, :]
        self.min = ts_obs.min(axis=0, keepdims=True)
        self.scale_normal = (ts_obs.max(axis=0, keepdims=True) - ts_obs.min(axis=0, keepdims=True))
        obs = (ts_obs - self.min) / self.scale_normal
        return obs

    def target(self, drop_open_volume=True):
        # ts_target = self.df[['high', 'low', 'close']].to_numpy()[-1, :].copy()
        ts_target = self.ts[-1, 1:-1]

        target = (ts_target - self.min[:, 1:-1]) / self.scale_normal[:, 1:-1]
        return target

    def prediction_to_ohlc_df(self, nn_prediction: np.ndarray):
        # inverse normalizer
        normal_prediction = (nn_prediction * self.scale_normal[:, 1:-1]) + self.min[:, 1:-1]
        p_df = self.ohlc_target.copy()
        # p_df[['high', 'low', 'close']] = normal_prediction.reshape(-1, len(normal_prediction))
        # if diff
        normal_prediction = normal_prediction.flatten()
        p_df['high'] = normal_prediction[0] + self.ohlc_obs['high'][-1]
        p_df['low'] = normal_prediction[1] + self.ohlc_obs['low'][-1]
        p_df['close'] = normal_prediction[2] + self.ohlc_obs['close'][-1]

        return p_df

#
# df = pd.read_csv('train/gold.csv', )  # , parse_dates=True)
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)
#
#
# input_to_predict = df.iloc[-3 * 3:].copy()
# proc = PrePostProcess(input_to_predict)
#
# nn_input_torch = proc.obs() # add dummy dim for batch
# # nn_output_numpy = MODEL ( nn_input_torch).numpy()
# nn_output_numpy = np.random.rand(3)  # proc.target()
# df_predict = proc.prediction_to_ohlc_df(nn_output_numpy)
# df_obs = proc.ohlc_obs
# df_target = proc.ohlc_target
#
# print('model_out',nn_output_numpy)
# print(df_predict)

