import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# https://stackoverflow.com/questions/47985835/tensorboard-is-not-recognized-as-an-internal-or-external-command
from model_loader import LitNetModel
from ny_data_loader import LitNyData
from models.RNN import LSTMModel
from models.CausalRnn import CausalRnn
from models.InceptionRaw import Inception

import pandas as pd
import pickle

# # READ DATA
df = pd.read_csv('smooth_gold.csv', )  # , parse_dates=True)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
#df = df.iloc[:1000,:]
# config = {}
with open('gold_config_CasualRnn.pkl', 'rb') as f:
    config = pickle.load(f)
    print(config)

logger = TensorBoardLogger("tb_logs", name="gold_model")
trainer = pl.Trainer(
    gpus=0,
    logger=logger,
    # max_epochs=10,
    # log_every_n_steps=50,
)
print(config)
config['learning_rate'] = 1e-3
# config['number_days'] = 20

if __name__ == '__main__':
    data_module = LitNyData(df, config)
    model = LitNetModel(CausalRnn, config)

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("best_model.ckpt")
