import torch
import torch.nn as nn
from models.CausalConv1d import CausalConv1d


class Inception(nn.Module):
    def __init__(self,
                 in_channels,
                 bottleneck_channels=32,
                 hidden_channels=16,
                 kernel_sizes=(2, 4, 8),
                 ):
        super(Inception, self).__init__()

        self.bottleneck = CausalConv1d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            bias=False
        )

        self.conv_from_bottleneck_1 = CausalConv1d(
            in_channels=bottleneck_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_sizes[0],
            bias=False
        )
        self.conv_from_bottleneck_2 = CausalConv1d(
            in_channels=bottleneck_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_sizes[1],
            bias=False
        )
        self.conv_from_bottleneck_3 = CausalConv1d(
            in_channels=bottleneck_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_sizes[2],
            bias=False
        )

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, )
        self.conv_from_max_pool = CausalConv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False
        )
        self.batch_norm = nn.BatchNorm1d(num_features=4 * hidden_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        # step 1
        z_bottleneck = self.bottleneck(x)
        z_max_pool = self.max_pool(x)
        # step 2
        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_max_pool(z_max_pool)
        # step 3
        z = torch.cat([z1, z2, z3, z4], dim=1)
        z = self.activation(self.batch_norm(z))

        return z


class InceptionBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_sizes,
                 bottleneck_channels,
                 dropout,
                 use_residual=True,
                 ):
        super(InceptionBlock, self).__init__()

        self.use_residual = use_residual
        self.activation = nn.ReLU()
        self.inception_1 = Inception(
            in_channels=in_channels,
            kernel_sizes=kernel_sizes[0],
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels
        )
        self.inception_2 = Inception(
            in_channels=4 * hidden_channels,  # in_channels,  #
            kernel_sizes=kernel_sizes[1],
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels

        )
        self.inception_3 = Inception(
            in_channels=4 * hidden_channels,  # in_channels,  #
            kernel_sizes=kernel_sizes[2],
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels

        )


    def forward(self, x):
        x1 = self.inception_1(x)
        # x2 = x1 + self.dropout1(self.inception_2(x))
        # x3 = x2 + self.dropout2(self.inception_3(x))
        x2 = self.inception_2(x1)  # self.dropout1(self.inception_2(x1))
        x3 = self.inception_3(x2)  # self.dropout2(self.inception_3(x2))
        # x = x + self.residual(x)
        x4 = self.activation(x3)

        return x4

# x = torch.rand(1, 4, 16)
#
# model = InceptionBlock(
#     in_channels=4,
#     bottleneck_channels=4,
#     hidden_channels=32,
#     kernel_sizes=(2, 4, 8)
# )
# # print(x.shape)
# out = model(x)
# print(out.shape)
