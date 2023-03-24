import pandas as pd
import numpy as np


class NyDiffNormalizer:
    def __init__(self,
                 df: pd.DataFrame
                 ):
        self.df = df
        #print(df)
        # 0-1 min-max Normal
        self.scale_normal = None
        self.min = None
        self.trend_speed = None
        self.ndf = self.ohlc_normal_df()
        self.x_df = self.ndf.copy()  # pd.concat([self.ndf,self.tech],axis=1)

    def obs(self):
        ts_obs = self.x_df.to_numpy().copy()

        return ts_obs

    def target(self,):
        ts_target = self.df[['high', 'low', 'close']].to_numpy()[-1, :].copy()
        ts_target -= self.min
        ts_target /= self.scale_normal
        return ts_target

    def ohlc_normal_df(self, ):
        d_df = self.df
        n_df = d_df[['open','high', 'low', 'close']].iloc[:-1, :].copy()
        #print(n_df)
        self.min = n_df.low.min()
        self.scale_normal = (n_df.high.max() - n_df.low.min())
        n_df = (n_df - self.min) / self.scale_normal
        return n_df



    def prediction_to_ohlc_df(self, nn_prediction: np.ndarray):
        # inverse normalizer
        normal_prediction = (nn_prediction * self.scale_normal) + self.min
        p_df = self.df.iloc[-1, :].copy()
        # p_df[['high', 'low', 'close']] = normal_prediction.reshape(-1, len(normal_prediction))
        # if diff
        normal_prediction = normal_prediction.flatten()
        p_df['high'] = normal_prediction[0]  # + self.ohlc_obs['high'][-1]
        p_df['low'] = normal_prediction[1]  # + self.ohlc_obs['low'][-1]
        p_df['close'] = normal_prediction[2]  # + self.ohlc_obs['close'][-1]

        return p_df


# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# # # # #
# xdf = pd.read_csv('smooth_gold.csv', )  # , parse_dates=True)
# xdf['time'] = pd.to_datetime(xdf['time'])
# xdf.set_index('time', inplace=True)
# ohlc = xdf.iloc[:3*40, :]
# nydiff = NyDiffNormalizer(ohlc)
#
# x=nydiff.obs()
# scaler = preprocessing.MinMaxScaler().fit(x)
# X_scaled = scaler.transform(x)
# plt.figure(1)
# plt.plot(x)
#
# plt.figure(2)
# plt.plot(X_scaled)
# plt.show()