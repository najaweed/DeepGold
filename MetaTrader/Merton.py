import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # READ DATA
df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
x=df.iloc[:180,:-1].mean(axis=1).to_numpy()

def time_series_embedding(data, delay=1, dimension=2):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay * dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')
    emd_ts = np.array([data[0:len(data) - delay * dimension]])
    for i in range(1, dimension):
        emd_ts = np.append(emd_ts, [data[i * delay:len(data) - delay * (dimension - i)]], axis=0)
    return emd_ts
emb_x = time_series_embedding(x,delay=2,dimension=8)
print(emb_x)
print(emb_x.shape)


def svd_smoother(emb_x,rank=1):
    u, s, v_h = np.linalg.svd(emb_x, full_matrices=False)
    u_r, s_r, v_h_r = u[:, 0:rank], s[0:rank], v_h[0:rank, :]
    xx = np.matmul(np.matmul(u_r, np.diag(s_r)), v_h_r)

    return xx[-1, :]
s_x = svd_smoother(emb_x)
plt.plot(x)
plt.plot(np.arange(len(x)-len(s_x),len(x)),s_x)
plt.show()