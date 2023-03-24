import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pre_process import NyDiffNormalizer

# class NyDiffNormalizer:
#     def __init__(self,
#                  df: pd.DataFrame
#                  ):
#         self.df = df#.drop(columns=['tick_volume'])
#         #print(self.df)
#         # print(np.diff(self.df.to_numpy(), axis=0))
#         self.ts = np.diff(self.df.to_numpy(), axis=0)  # self.df  # np.diff(self.df.to_numpy(), axis=0)
#         self.scale_normal = None
#         self.min = None
#         # self.obs = self.obs()
#         # self.target = self.target()
#
#     def obs(self):
#         # ts_obs = self.ts.to_numpy()[:-1, :]
#         ts_obs = self.ts[:-1, :]
#
#         # print(ts_obs)
#
#         self.min = ts_obs.min(axis=0, keepdims=True)
#         self.scale_normal = (ts_obs.max(axis=0, keepdims=True) - ts_obs.min(axis=0, keepdims=True))
#         obs = (ts_obs - self.min) / self.scale_normal
#         # print(obs)
#         return obs
#
#     def target(self, drop_open_volume=True):
#         # ts_target = self.df[['high', 'low', 'close']].to_numpy()[-1, :].copy()
#         ts_target = self.ts[-1, 1:-1]
#         # print(ts_target)
#         # target = (ts_target - self.min) / self.scale_normal
#
#         target = (ts_target - self.min[:, 1:-1]) / self.scale_normal[:, 1:-1]
#         # print(target.shape)
#         target = np.clip(target,a_min=0,a_max=1.0)
#         return target
#

class NyDataset(Dataset):
    def __init__(self,
                 data_temporal: pd.DataFrame,
                 config: dict,
                 ):
        self.time_series = data_temporal
        self.step_predict = 1
        self.step_share = 0
        self.tick_per_day = config['tick_per_day']
        self.num_days = config['number_days']
        self.window_temporal = self.tick_per_day * self.num_days
        self.obs, self.target = self.split_observation_prediction()

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):
        return self.obs[index, ...], self.target[index]

    def split_observation_prediction(self):
        x, y = [], []
        # print(self.time_series)
        in_shape, target_shape = None, None
        for t in range(0, len(self.time_series) - self.window_temporal + self.tick_per_day, self.tick_per_day):

            x_df = self.time_series.iloc[t:(t + self.window_temporal), :].copy()
            # print(x_df)
            ny_normal = NyDiffNormalizer(x_df)
            obs = ny_normal.obs()
            target = ny_normal.target()
            if t == 0:
                in_shape = obs.shape
                target_shape = target[0].shape
            # print(obs.shape,target.shape)

            if target_shape == target_shape and obs.shape == in_shape:
                x.append(obs)
                y.append(target)
        # print(x , y)
        # x, y = np.float32(np.array(x, dtype=object)), np.float32(np.array(y, dtype=object))
        # print(x.shape, y.shape)
        # print(x, y)

        # breakpoint()
        x = torch.tensor(np.array(x, dtype=np.float32), dtype=torch.float32)
        y = torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float32)
        return x, y


class LitNyData(pl.LightningDataModule, ):

    def __init__(self,
                 df: pd.DataFrame,
                 config: dict,
                 ):
        super().__init__()
        self.config = config
        self.df = df
        self.train_loader, self.val_loader = self._gen_data_loaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def _gen_data_loaders(self):
        split_index = [int(self.df.shape[0] * self.config['split'][i] / 10) for i in range(len(self.config['split']))]
        start_index = 0
        # time_series = torch.FloatTensor(self.df[[f'{sym}.si,close' for sym in self.symbols]].values)
        data_loaders = []
        for i in range(len(self.config['split'])):
            end_index = split_index[i] + start_index
            dataset = NyDataset(self.df.iloc[start_index:end_index, :], self.config)
            data_loaders.append(DataLoader(dataset=dataset,
                                           batch_size=self.config['batch_size'],
                                           # num_workers=4,
                                           drop_last=True,
                                           # pin_memory=True,
                                           shuffle=False,

                                           ))
            start_index = end_index
        return data_loaders


