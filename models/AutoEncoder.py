import torch
import torch.nn as nn
from models.InceptionTimeX import InceptionBlock, InceptionTransposeBlock


class VariationEncoder(torch.nn.Module):
    def __init__(self,
                 config,
                 ):
        super(VariationEncoder, self).__init__()
        self.encoder = InceptionBlock(in_channels=config['in_channels'],
                                      n_filters=config['hidden_channels'],
                                      bottleneck_channels=config['bottleneck_channels'],
                                      return_indices=True)

        self.fc_var = torch.nn.Linear(in_features=4 * config['hidden_channels'],
                                      out_features=4 * config['hidden_channels'])

    def forward(self, x):
        if self.training:
            h, index = self.encoder(x)
            print(h.shape)
            logvar = self.fc_var(torch.permute(h, (0, 1, 2)))
            z = self._reparameterize(h, logvar)
            # y = self.decoder(z)
            return z, h, logvar, index
        else:
            z = self.represent(x)
            # y = self.decoder(z)
            return z

    def represent(self, x):
        h, index = self.encoder(x)
        mu, logvar = h, self.fc_var(h)
        z = self._reparameterize(mu, logvar)
        return z, index

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).type_as(mu)
        z = mu + std * esp
        return z


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.inception = InceptionBlock(in_channels=config['in_channels'],
                                        n_filters=config['hidden_channels'],
                                        bottleneck_channels=config['bottleneck_channels'],
                                        return_indices=True)

    def forward(self, x):
        x = self.inception(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.trans_inception = InceptionTransposeBlock(in_channels=4 * config['hidden_channels'],
                                                       out_channels=config['in_channels'],
                                                       bottleneck_channels=config['bottleneck_channels'])

    def forward(self, z):
        x = self.trans_inception(*z)
        return x


class Autoencoder(nn.Module):
    def __init__(self, config):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):
        z = self.encoder(x)
        # print(z[0].shape)
        return self.decoder(z)


xx = torch.rand(1, 4, 120)
x_config = {
    'in_channels': 4,
    'bottleneck_channels': 32,
    'hidden_channels': 1,
}
model = VariationEncoder(x_config)
xs = model(xx)
print(xs.shape)
