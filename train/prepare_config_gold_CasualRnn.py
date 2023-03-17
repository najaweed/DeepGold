from ny_data_loader import LitNyData

import pandas as pd

# # READ DATA
df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
# # df.drop('time', axis=1, inplace=True)
# df = df.dropna(axis=0, how='any')
# df = df.diff().dropna(axis=0, how='any').ewm(alpha=0.1).mean()
# READ DATA
config_data_loader = {
    # config dataset and dataloader
    'batch_size': 32,
    'tick_per_day': 3,
    'number_days': 80,
    'split': (7, 2, 1),  # make a function for K-fold validation
}
# FIND MODEL CONFIG BASED ON DATALOADER
lit_data = LitNyData(df, config_data_loader)
lit_val = lit_data.val_loader
in_shape, out_shape = None, None

config_CasualRnn = {
    'bottleneck_channels': 128,
    'hidden_channels': 32,
    'kernel_sizes': (2, 4, 8),
    'num_stack_layers': 1,
    'dropout': 0.4,
}
in_sample = None

for i, (a, b) in enumerate(lit_data.train_dataloader()):
    in_sample = a
    in_shape, out_shape = a.size(), b.size()
    config_CasualRnn['in_channels'] = in_shape[-1]
    config_CasualRnn['output_size'] = out_shape[-1] * out_shape[-2]
    print(a)
    print(b)
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