import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    '''
    shape : [input] = [ batch , seq_time, in_channel_size] ===>
    [out] = [ batch, output_size]
    '''

    def __init__(self,
                 config: dict,
                 ):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size=config['in_channel_size'],
                            hidden_size=config['hidden_size_channels'],
                            num_layers=config['num_stack_layers'],
                            dropout=config['dropout'],
                            batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(config['num_stack_layers'] * config['hidden_size_channels'],
                                  out_features=config['output_size'])

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batch_size = x.shape[0]
        # LSTM layer
        h0 = torch.zeros(self.config['num_stack_layers'], batch_size,
                         self.config['hidden_size_channels'], requires_grad=True)  # .to(device)
        c0 = torch.zeros(self.config['num_stack_layers'], batch_size,
                         self.config['hidden_size_channels'], requires_grad=True)  # .to(device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h0, c0))
        # reshape output from hidden cell into [batch, features] for `linear_2`
        # print(h_n.permute(1, 0, 2).shape)
        x = h_n.permute(1, 0, 2).reshape(batch_size, -1)
        # layer 2
        x = self.dropout(x)
        x = self.relu(x)
        # print(x.shape)
        predictions = self.linear_2(x)
        return predictions  # [:, -1:]

# config = dict(
#     num_stack_layers=3,
#     hidden_size_channels=17,
#     batch=8,
#     in_channel_size=1,
#     seq_len=6,
#     dropout=0.2,
#     output_size=3,
# )
# x = torch.ones(config['batch'], config['seq_len'], config['in_channel_size'])
#
#
# model = LSTMModel(config)
# print(model(x).shape)
