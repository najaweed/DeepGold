import torch
import torch.nn as nn


class CnnSimpleNet(nn.Module):
    def __init__(self,
                 config: dict,
                 ):
        super(CnnSimpleNet, self).__init__()
        self.first_layer = torch.nn.Sequential(
            nn.ConvTranspose2d(config['in_channels'],
                               config['hidden_channels'],
                               kernel_size=config['kernel_first'],
                               bias=False),
            nn.Tanh(), )

        self.last_layer = torch.nn.Sequential(
            nn.Conv2d(config['hidden_channels'],
                      config['out_channels'],
                      kernel_size=config['kernel_last'],
                      bias=False),
            nn.Tanh(), )

    def forward(self, x):
        x = self.first_layer(x)
        # print('hidedn_shape',x.shape)
        x = self.last_layer(x)
        return x


#
#
# def calculate_kernel_first_layer(sample_input, shape_first_layer=(16, 16)):
#     kernel = [0, 0]
#     shape_input = sample_input.size()
#
#     for h in range(1, shape_first_layer[-1]):
#         # print('h', h)
#         model = nn.ConvTranspose2d(1,
#                                    1,
#                                    kernel_size=(shape_input[-2], h),
#                                    bias=False)
#         # print(shape_input[-i])
#         if shape_first_layer[-1] > shape_input[-1]:
#             # calculate kernel
#             x = model(sample_input)
#             if x.shape[-1] == shape_first_layer[-1]:
#                 # print(x.shape)
#
#                 kernel[-1] = h
#                 break
#         elif shape_first_layer[-1] == shape_input[-1]:
#             kernel[-1] = shape_input[-1]
#
#         for w in range(1, shape_first_layer[-2]):
#             # print('h', w)
#             model = nn.ConvTranspose2d(1,
#                                        1,
#                                        kernel_size=(w, shape_input[-1]),
#                                        bias=False)
#             # print(shape_input[-i])
#             if shape_first_layer[-2] > shape_input[-2]:
#                 # calculate kernel
#                 x = model(sample_input)
#                 if x.shape[-2] == shape_first_layer[-2]:
#                     # print(x.shape)
#
#                     kernel[-2] = w
#                     break
#             elif shape_first_layer[-2] == shape_input[-2]:
#                 kernel[-2] = shape_input[-2]
#     return kernel
#
#
# def calculate_kernel_last_layer(sample_input, shape_last_layer=(32, 32)):
#     kernel = [0, 0]
#     shape_input = sample_input.size()
#
#     for h in range(1, 100):
#         # print('H', h)
#         model = nn.Conv2d(1,
#                           1,
#                           kernel_size=(shape_input[-2], h),
#                           bias=False)
#         # print(shape_input[-i])
#         if shape_last_layer[-1] < shape_input[-1]:
#             # calculate kernel
#             x = model(sample_input)
#             if x.shape[-1] == shape_last_layer[-1]:
#                 # print(x.shape)
#
#                 kernel[-1] = h
#                 break
#         elif shape_last_layer[-1] == shape_input[-1]:
#             kernel[-1] = shape_input[-1]
#
#         for w in range(1, 100):
#             # print('W', w)
#             model = nn.Conv2d(1,
#                               1,
#                               kernel_size=(w, shape_input[-1]),
#                               bias=False)
#             # print(shape_input[-i])
#             if shape_last_layer[-2] < shape_input[-2]:
#                 # calculate kernel
#                 x = model(sample_input)
#                 if x.shape[-2] == shape_last_layer[-2]:
#                     # print(x.shape)
#
#                     kernel[-2] = w
#                     break
#             elif shape_last_layer[-2] == shape_input[-2]:
#                 kernel[-2] = shape_input[-2]
#
#     model = nn.Conv2d(1,
#                       1,
#                       kernel_size=kernel,
#                       bias=False)
#     # print('iaaa', model(sample_input).shape)
#     return kernel

#
# xinput = torch.randn(128, 1, 13, 6)
# xxinput = torch.randn(128, 1, 16, 16)
#
# xkernel = calculate_kernel_first_layer(xinput, [16, 16])
# xxkernel = calculate_kernel_last_layer(xxinput, [2, 6])
#
# print('in_kernel', xkernel)
# print('out_kernel', xxkernel)
#
# print('in_shape ', xinput.shape)
# xconfig = {
#     # config dataset and dataloader
#     'batch_size': 5,
#     'num_nodes': 6,
#     'freq_obs': '2W',
#     'step_predict': 2,
#     'step_share': 0,
#     'split': (7, 2, 1),
#     # config network
#     # config conv layers
#     'in_channels': 1,
#     'hidden_channels': 256,
#     'out_channels': 1,
#     'kernel_first': xkernel,
#     'kernel_last': xxkernel,
#
#     'num_layers': 8,
#     # config linear layer
#     'in_features': 8 * 6,
#     'hidden_features': 512,
#     'out_features': 2 * 6,
#     # config trainer
#     'learning_rate': 5e-2,
#
# }
#
# model = CnnSimpleNet(xconfig)
#
# output = model(xinput)
# # print(output)
# print('out_shape ', output.shape)
# plt.figure(1)
# plt.imshow(torch.flatten(xinput, end_dim=-2).detach().numpy())
# plt.figure(11)
# plt.imshow(torch.flatten(output, end_dim=-2).detach().numpy())
# # plt.show()
