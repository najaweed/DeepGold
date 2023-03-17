import torch
import torch.nn as nn
from LitTorch.models.InceptionTime import InceptionBlock


class CausalRnn(nn.Module):
    def __init__(self,
                 config: dict,
                 ):
        super(CausalRnn, self).__init__()
        self.config = config
        self.inception = InceptionBlock(
            in_channels=config['in_channels'],
            bottleneck_channels=config['bottleneck_channels'],
            hidden_channels=config['hidden_channels'],
            kernel_sizes=config['kernel_sizes'],
            dropout=config['dropout'],
        )
        self.lstm = nn.LSTM(input_size=4 * config['hidden_channels'],
                            hidden_size=4 * config['hidden_channels'],
                            num_layers=config['num_stack_layers'],
                            batch_first=True)
        self.dropout = nn.Dropout(config['dropout'])
        self.dropout0 = nn.Dropout(config['dropout'])
        self.linear = nn.Linear(in_features=4 * config['hidden_channels'],
                                out_features=config['output_size'])
        #self.linear = nn.LazyLinear(out_features=config['output_size'])
        #self.activ = nn.ReLU()
    #     self.init_weights()
    #
    # def init_weights(self):
    #     for name, param in self.lstm.named_parameters():
    #         if 'bias' in name:
    #             nn.init.constant_(param, 0.0)
    #         elif 'weight_ih' in name:
    #             nn.init.kaiming_normal_(param)
    #         elif 'weight_hh' in name:
    #             nn.init.orthogonal_(param)

    def forward(self, x):
        batch_size = x.shape[0]
        # Inception
        x = torch.permute(x, (0, 2, 1))
        x = self.inception(x)
        x = self.dropout0(x)
        x = torch.permute(x, (0, 2, 1))  # to [batch , seq_time , channels]
        # Recurrent
        h0 = torch.zeros(self.config['num_stack_layers'], batch_size,
                         4 * self.config['hidden_channels'], ).requires_grad_()

        c0 = torch.zeros(self.config['num_stack_layers'], batch_size,
                         4 * self.config['hidden_channels'], ).requires_grad_()
        lstm_out, (h_n, c_n) = self.lstm(x, (h0.detach(), c0.detach()))
        # Predictor
        x = self.dropout(h_n[-1])
        # batch_norm ?
        x = self.linear(x)
        # x = self.activ(x)
        return x

#
# x_in = torch.rand(1, 18, 4)
# x_config = {
#     'in_channels': 4,
#     'bottleneck_channels': 1,
#     'hidden_channels': 64,
#     'kernel_sizes': (2, 4, 8),
#     'num_stack_layers': 2,
#     'dropout': 0,
#     'output_size': 3,
# }
# model = CausalRnn(x_config)
# out = model(x_in)
# print(out.shape)
