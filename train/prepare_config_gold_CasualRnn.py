from ny_data_loader import LitNyData

import pandas as pd

# # READ DATA
df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df = df.iloc[:1000, :]
# # df.drop('time', axis=1, inplace=True)
# df = df.dropna(axis=0, how='any')
# df = df.diff().dropna(axis=0, how='any').ewm(alpha=0.1).mean()
# READ DATA
config_data_loader = {
    # config dataset and dataloader
    'batch_size': 1,
    'tick_per_day': 3,
    'number_days': 20,
    'split': (9, 1),  # make a function for K-fold validationb
}
# FIND MODEL CONFIG BASED ON DATALOADER
lit_data = LitNyData(df, config_data_loader)
lit_val = lit_data.val_loader
in_shape, out_shape = None, None

config_CasualRnn = {
    'bottleneck_channels': 8,
    'hidden_channels': 2,
    'kernel_sizes': ((10, 20, 40), (10, 20, 40), (10, 20, 40)),
    'num_stack_layers': 1,
    'dropout': 0.4,
    'kernel_avg':4,
}
in_sample = None

for i, (a, b) in enumerate(lit_data.train_dataloader()):
    in_sample = a
    in_shape, out_shape = a.size(), b[0].size()
    config_CasualRnn['in_channels'] = in_shape[-1]
    config_CasualRnn['output_size'] = out_shape[-1]  # * out_shape[-2]
    print(a.shape)
    print(b[0].shape)
    print('input_shape ', in_shape, ',', 'output_shape ', out_shape)
    print(i)
    # kernel = calculate_kernel_last_layer(a, out_shape)
    # kernel_last_layer = kernel
    break
config = config_data_loader | config_CasualRnn  # == {**config_data_loader , **config_conv}
print(config)
import pickle

with open('gold_config_CasualRnn.pkl', 'wb') as f:
    pickle.dump(config, f)
