import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('train/gold.csv', index_col='time', parse_dates=True)
print(df)

s_df = df.iloc[:, :-1].ewm(alpha=0.8).mean()
plt.figure(1)
for col  in ['open','high','low','close']:
    print(col)
    print(np.mean(abs(df[col] - s_df[col])))
    m_col = np.mean(abs(df[col] - s_df[col]))
    #plt.plot(df[col]-s_df[col],c='r')

    for i in range(len(df)):
        if abs(df[col][i] - s_df[col][i]) > 5:
            print(i)
            #plt.axvline(i, c='b')
            print(df[col][i])
            df[col][i] += s_df[col][i]
            df[col][i] /=2
            print(df[col][i])
    #plt.plot(df[col]-s_df[col])
    #plt.show()

print(df)
df.to_csv('smooth_gold.csv',index_label='time')
