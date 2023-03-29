import torch
import torch.nn as nn
import pickle
from models.AutoEncoder import Autoencoder
from models.CausalConv1d import CausalConv1d


class EncoderConv(nn.Module):
    def __init__(self, config):
        super(EncoderConv, self).__init__()
        self.config = config
        self.encoder = self.load_encoder(config)
        self.seq_conv_1 = nn.Sequential(
            CausalConv1d(in_channels=4 * config['hidden_channels'],
                         out_channels=8 * config['hidden_channels'],
                         kernel_size=2,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=8 * config['hidden_channels'],
                         out_channels=8 * config['hidden_channels'],
                         kernel_size=4,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=8 * config['hidden_channels'],
                         out_channels=1,
                         kernel_size=8,
                         bias=False),
            nn.ReLU(),
        )
        self.seq_conv_2 = nn.Sequential(
            CausalConv1d(in_channels=4 * config['hidden_channels'],
                         out_channels=8 * config['hidden_channels'],
                         kernel_size=2,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=8 * config['hidden_channels'],
                         out_channels=8 * config['hidden_channels'],
                         kernel_size=4,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=8 * config['hidden_channels'],
                         out_channels=1,
                         kernel_size=8,
                         bias=False),
            nn.ReLU(),
        )
        self.seq_conv_3 = nn.Sequential(
            CausalConv1d(in_channels=4 * config['hidden_channels'],
                         out_channels=8 * config['hidden_channels'],
                         kernel_size=2,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=8 * config['hidden_channels'],
                         out_channels=8 * config['hidden_channels'],
                         kernel_size=4,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=8 * config['hidden_channels'],
                         out_channels=1,
                         kernel_size=8,
                         bias=False),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(config['dropout'])
        self.batch_norm = nn.BatchNorm1d(4 * config['hidden_channels'])

    @staticmethod
    def load_encoder(config, path: str = "encoder_params.ckpt"):
        model = Autoencoder(config)
        model_weights = torch.load(path)["state_dict"]
        # update keys by dropping `nn_model.`
        for key in list(model_weights):
            model_weights[key.replace("nn_model.", "")] = model_weights.pop(key)
        model.load_state_dict(model_weights)
        model.eval()
        x_encoder = model.encoder
        return x_encoder

    def forward(self, x):
        # Encoder Inception
        with torch.no_grad():
            x, _ = self.encoder(x.float())
        x = self.batch_norm(x)
        x = self.dropout(x)
        x1 = self.seq_conv_1(x)
        x2 = self.seq_conv_2(x)
        x3 = self.seq_conv_3(x)
        x = torch.concat([x1, x2, x3], dim=1)
        return x[:, :, -1]

#
# with open('gold_config_CasualRnn.pkl', 'rb') as f:
#     xconfig = pickle.load(f)
#     print(xconfig)
#
# from pre_process import NyDiffNormalizer
# import pandas as pd
# import numpy as np
#
# df = pd.read_csv('gold.csv', )  # , parse_dates=True)
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)
# # print(df)
#
# window = xconfig['number_days'] * xconfig['tick_per_day']
# input_to_predict = df.iloc[-13 * window:-12 * window].copy()
# # print(input_to_predict)
# # nydiff = NyDiffNormalizer(ohlc)
# proc = NyDiffNormalizer(input_to_predict.iloc[:, :])
# nn_input_numpy = proc.obs()  # add dummy dim for batch
# nn_input_torch = torch.from_numpy(np.expand_dims(nn_input_numpy, 0))
#
# # print(nn_input_torch)
# nn_input_torch = torch.permute(nn_input_torch, (0, 2, 1))
# model = EncoderConv(xconfig)
# print(model(nn_input_torch).shape)
