import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self,
                config:dict,
                 ):
        super(SimpleNet, self).__init__()
        print(config)
        self.lin1 = nn.Linear(in_features=config['in_features'], out_features=config['hidden_features'])
        self.relu = nn.ReLU()
        self.lin3 = nn.Linear(in_features=config['hidden_features'], out_features=config['hidden_features'])

        self.lin2 = nn.Linear(in_features=config['hidden_features'], out_features=config['out_features'])
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.tanh(x)

        return x

# config = {
#     'batch_size': 1,
#     'num_nodes': 6,
#     'freq_obs': '3W',
#     'step_predict': 3,
#     'step_share': 0,
#     'split': (7, 2, 1),
#     'in_channels': 2 * 3 * 4,
#     'hidden_channels': 256,
#     'out_channels': 2 * 3,
#     'num_layers': 8,
#     'learning_rate': 5e-3,
#     'in_features': 12 * 6,
#     'hidden_features': 128,
#     'out_features': 3 * 6,
# }
#model = SimpleNet(config)
#input = torch.ones(1, 1, 12, 6)
#print(input)
#print(input.shape)
#output = model(input)
#print(output)
#print(output.shape)
