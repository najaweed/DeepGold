import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# https://stackoverflow.com/questions/47985835/tensorboard-is-not-recognized-as-an-internal-or-external-command
from model_loader import LitNetModel
from ny_data_loader import LitNyData
from models.AutoEncoder import Autoencoder

import pandas as pd
import pickle

# # READ DATA
df = pd.read_csv('gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df = df.loc['2019-01-01':]
with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)

logger = TensorBoardLogger("tb_logs", name="gold_model")
trainer = pl.Trainer(
    # gpus=0,
    logger=logger,
    max_epochs=60,
    #log_every_n_steps=5,
)
print(config)
config['learning_rate'] = 1e-3

if __name__ == '__main__':
    data_module = LitNyData(df, config)
    model = LitNetModel(Autoencoder, config, is_encoder=True)

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("encoder_params.ckpt")
