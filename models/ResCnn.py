import torch
import torch.nn as nn

from LitTorch.models.deep_residual_network import DeepGCNLayer


class ResCnn(nn.Module):
    def __init__(self,
                 config: dict,
                 ):
        super(ResCnn, self).__init__()

        self.first_layer = torch.nn.Sequential(
            nn.Conv2d(config['in_channels'],
                      config['hidden_channels'],
                      kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2),
                      bias=False),
            #    nn.Tanh(),
        )

        self.residual_block = nn.ModuleList()
        for i in range(1, config['num_res_layers']):
            conv = nn.Conv2d(config['hidden_channels'], config['hidden_channels'],
                             kernel_size=(3, 3), padding=(2, 2), dilation=(2, 2))
            norm = nn.BatchNorm2d(config['hidden_channels'], )
            act = nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res',)# dropout=0.4, )
            self.residual_block.append(layer)

        self.last_layer = torch.nn.Sequential(
            nn.Conv2d(config['hidden_channels'],
                      config['out_channels'],
                      kernel_size=config['kernel_last'],
                      bias=False),
        #    nn.Tanh(),
        )

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.residual_block.children():
            x = layer(x)
        x = self.last_layer(x)
        return x

# from models.ResCnn import ResCnn
# model = ResCnn(config_conv)
# x = model(sample_input)
# print(x.shape)
# print(model)
# print(sample_input.shape)
# print(x)
# from torchinfo import summary
#
# summary(model, input_size=sample_input.shape)
