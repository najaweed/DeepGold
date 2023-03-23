import pandas as pd
import numpy as np


class NyDiffNormalizer:
    def __init__(self,
                 df: pd.DataFrame
                 ):
        self.df = df
        # 0-1 min-max Normal
        self.scale_normal = None
        self.min = None
        self.trend_speed = None
        self.ndf = self.ohlc_normal_df()
        # self.tech = self.technical_indicators(window_temporal=15)
        self.x_df = self.ndf.copy()  # pd.concat([self.ndf,self.tech],axis=1)

    def obs(self):
        ts_obs = self.x_df.to_numpy().copy()[:-1, :]
        return ts_obs

    def target(self, ):
        ts_target = self.df[['high', 'low', 'close']].to_numpy()[-1, :].copy()
        # print(self.df[['open']].to_numpy()[-1, :].copy()[0])
        return ts_target, self.scale_normal, self.df[['open']].to_numpy()[-1, :].copy()[0] + self.trend_speed

    def ohlc_normal_df(self, ):
        d_df = self.df_de_trend(self.df)
        n_df = d_df.iloc[:-1, :].copy()
        self.min = n_df.low.min()
        self.scale_normal = (n_df.high.max() - n_df.low.min())
        n_df = (n_df - self.min) / self.scale_normal
        return n_df


    def df_de_trend(self,in_df):
        x = in_df[['open', 'high', 'low', 'close']].mean(axis=1).to_numpy()
        #print(x)
        y = np.arange(0, len(x))
        m_b = np.polyfit(y, x, 1)
        m, b = m_b[0], m_b[1]
        self.trend_speed = m
        #print(m, b)
        y_hat = m * y + b
        x_df = in_df[['open', 'high', 'low', 'close']].copy()
        # x_df['y_hat'] = y_hat
        # print(x_df['y_hat'])
        #print(len(in_df))
        #print(len(y_hat))
        #print(len(x))
        return x_df.sub(y_hat, axis='index')

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

    def technical_indicators(self, window_temporal: int = 10):
        df_tech = pd.DataFrame()
        df_tech['STO'] = self.stochastic_oscillator_d(window_temporal)
        # df_tech['ACD'] = self.accumulation_distribution(window_temporal)
        df_tech['TSI'] = self.true_strength_index(int(window_temporal / 2), window_temporal)
        df_tech['MSI'] = self.mass_index()
        df_tech['FRI'] = self.force_index(window_temporal)
        df_tech['EOM'] = self.ease_of_movement(window_temporal)
        df_tech['CHI'] = self.chaikin_oscillator()
        df_tech = (df_tech - df_tech.min()) / (df_tech.max() - df_tech.min())
        # print(df_tech)
        return df_tech

    def stochastic_oscillator_d(self, n: int = 20):
        """Calculate stochastic oscillator %D for given data.
        """
        df = self.df.copy()
        SOk = (df['close'] - df['low']) / (df['high'] - df['low'])
        SOd = SOk.ewm(span=n, min_periods=n).mean()
        return SOd.fillna(method='bfill')

    def mass_index(self):
        """Calculate the Mass Index for given data.
        """
        df = self.df.copy()
        Range = df['high'] - df['low']
        EX1 = Range.ewm(span=9, min_periods=9).mean()
        EX2 = EX1.ewm(span=9, min_periods=9).mean()
        Mass = EX1 / EX2
        MassI = Mass.rolling(25).sum()
        # df = df.join(MassI)
        return MassI.fillna(method='bfill')

    def true_strength_index(self, r=10, s=20):
        """Calculate True Strength Index (TSI) for given data.
        """
        df = self.df.copy()
        M = pd.Series(df['close'].diff(1))
        aM = abs(M)
        EMA1 = M.ewm(span=r, min_periods=r).mean()
        aEMA1 = aM.ewm(span=r, min_periods=r).mean()
        EMA2 = EMA1.ewm(span=s, min_periods=s).mean()
        aEMA2 = aEMA1.ewm(span=s, min_periods=s).mean()
        TSI = EMA2 / aEMA2
        # df = df.join(TSI)
        return TSI.fillna(method='bfill')

    def accumulation_distribution(self, n=20):
        """Calculate Accumulation/Distribution for given data.
        """
        df = self.df.copy()
        ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['tick_volume']
        M = ad.diff(n - 1)
        N = ad.shift(n - 1)
        ROC = M / N
        return ROC.fillna(method='bfill')

    def force_index(self, n=20):
        """Calculate Force Index for given data.
        """
        df = self.df.copy()
        F = df['close'].diff(n) * df['tick_volume'].diff(n)
        # df = df.join(F)
        return F.fillna(method='bfill')

    def ease_of_movement(self, n=20):
        """Calculate Ease of Movement for given data.
        """
        df = self.df.copy()

        EoM = (df['high'].diff(1) + df['low'].diff(1)) * (df['high'] - df['low']) / (2 * df['tick_volume'])
        Eom_ma = EoM.rolling(n, min_periods=n).mean()
        # df = df.join(Eom_ma)
        return Eom_ma.fillna(method='bfill')

    def chaikin_oscillator(self, r=10, s=20):
        """Calculate Chaikin Oscillator for given data.
        """
        df = self.df.copy()

        ad = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['tick_volume']
        Chaikin = ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean()
        # df = df.join(Chaikin)
        return Chaikin.fillna(method='bfill')


import matplotlib.pyplot as plt

# # # #
# xdf = pd.read_csv('smooth_gold.csv', )  # , parse_dates=True)
# xdf['time'] = pd.to_datetime(xdf['time'])
# xdf.set_index('time', inplace=True)
# ohlc = xdf.iloc[-6 * 20:-1 * 20, :]
# nydiff = NyDiffNormalizer(ohlc)
# # print(nydiff.obs().shape)
# # print(nydiff.prediction_to_ohlc_df(np.array([0.1,0.5,0.2])))
# print(nydiff.scale_normal)
#
