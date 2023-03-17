import mplfinance as mpf
import pandas as pd
import finplot as fplt

df = pd.read_csv('train/gold.csv', index_col=0, parse_dates=True)
df.index.name = 'Date'
df.rename({'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, axis='columns')
xdf = df.iloc[-30:, :]
mpf.plot(xdf)
xdf = df.iloc[-40:-30, :]
mpf.plot(xdf, type='candle')
mpf.show()


class PlotPrediction:
    def __init__(self,
                 df_obs_ohlc: pd.DataFrame,
                 df_target_ohlc: pd.DataFrame,
                 df_prediction_ohlc: pd.DataFrame,
                 ):
        self.obs = df_obs_ohlc
        self.target = df_target_ohlc
        # self.prediction = df_prediction_hlc
        # self.prediction['open'] = self.target['open'].copy()
