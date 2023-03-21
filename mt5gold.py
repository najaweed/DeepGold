from datetime import datetime, timedelta
import time
import pytz
import pandas as pd
import numpy as np
import MetaTrader5 as mt5

#from mt5 import ForexMarket

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()


def get_last_day(symbol: str, shift_5m=400, s=0):
    ticks = pd.DataFrame(mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M30, s, shift_5m))
    ticks['time'] = pd.to_datetime(ticks['time'], unit='s')
    ticks = ticks.set_index('time')
    return ticks


xau = get_last_day('XAUUSD', shift_5m=99999, s=0)


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


def group_data(freq, x_df):
    df = x_df.copy()
    df['time'] = df.index
    df_list = []
    for group_name, df_group in df.groupby(pd.Grouper(freq=freq, key='time')):
        g_df = df_group.drop(columns=['time'])
        # week_or_month = 20 if self.freq[1] == 'M' else 5
        if len(g_df) == 3:  # 5days*5weeks
            df_list.append(g_df)
    return pd.DataFrame(pd.concat(df_list, axis=0).sort_index())


s_sessions = [('00:01', '7:00'), ('07:01', '13:00'), ('13:01', '23:59')]
t_df = session_transform(sessions=s_sessions, x_df=xau)
# print(t_df.to_numpy())
# import numpy as np

# print(np.diff(t_df.to_numpy(), axis=0))
xx_df = group_data('1D', t_df)
print(xx_df)

xx_df.to_csv('gold.csv',index_label='time')
