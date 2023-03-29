import torch
import torch.nn as nn
import pickle
from models.AutoEncoder import Autoencoder
from models.CausalConv1d import CausalConv1d


class EncoderLSTM(nn.Module):
    def __init__(self, config):
        super(EncoderLSTM, self).__init__()
        self.config = config
        self.encoder = self.load_encoder(config)
        self.lstm = nn.LSTM(input_size=4 * config['hidden_channels'],
                            hidden_size=4 * config['hidden_channels'],
                            num_layers=config['num_stack_layers'],
                            batch_first=True)

        self.temp_conv = nn.Sequential(
            CausalConv1d(in_channels=4 * config['hidden_channels'],
                         out_channels=config['hidden_channels'],
                         kernel_size=2,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=config['hidden_channels'],
                         out_channels=config['hidden_channels'],
                         kernel_size=2,
                         bias=False),
            nn.ReLU(),
            # nn.Dropout(),
            CausalConv1d(in_channels=config['hidden_channels'],
                         out_channels=int(config['hidden_channels'] / 2),
                         kernel_size=4,
                         bias=False),
            nn.ReLU(),
            CausalConv1d(in_channels=int(config['hidden_channels'] / 2),
                         out_channels=int(config['hidden_channels'] / 2),
                         kernel_size=4,
                         bias=False),
            nn.ReLU(),
            # nn.Dropout(),
            CausalConv1d(in_channels=int(config['hidden_channels'] / 2),
                         out_channels=config['output_size'],
                         kernel_size=8,
                         bias=False),

        )
        self.linear_1 = nn.Linear(in_features=4 * config['hidden_channels'],
                                  out_features=2 * config['hidden_channels'])
        self.linear_2 = nn.Linear(in_features=2 * config['hidden_channels'],
                                  out_features=config['output_size'])
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()
        self.lazy_batch_norm = nn.BatchNorm1d(4 * config['hidden_channels'])

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

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
        x = self.dropout(x)
        x = torch.permute(x, (0, 2, 1))  # to [batch , seq_time , channels]
        #x = self.lazy_batch_norm(x)
        #x = self.temp_conv(x)
        # # Recurrent Predictor
        #x = self.dropout(x)
        #x = self.lazy_batch_norm(x)
        h0 = torch.zeros(self.config['num_stack_layers'], x.shape[0],
                         4*self.config['hidden_channels'], ).requires_grad_()
        c0 = torch.zeros(self.config['num_stack_layers'], x.shape[0],
                         4*self.config['hidden_channels'], ).requires_grad_()
        lstm_out, (h_n, c_n) = self.lstm(x, (h0.detach(), c0.detach()))

        x = self.linear_1(h_n[-1])
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear_2(x)
        # print(x[:,:,-1].shape)
        # x = self.linear_2(x[:, :, -1])
        return x
#
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
# model = EncoderLSTM(xconfig)
# print(model(nn_input_torch))
