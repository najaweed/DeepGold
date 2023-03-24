import torch
import pytorch_lightning as pl


class LitNetModel(pl.LightningModule, ):

    def __init__(self,
                 net_model,
                 config: dict,
                 ):
        super().__init__()

        # configuration
        self.lr = config['learning_rate']
        # model initialization
        self.nn_model = net_model(config)
        self.loss = torch.nn.MSELoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # prepare inputs
        x = train_batch[0]
        #target = torch.flatten(train_batch[1], start_dim=1)
        # process model
        x = self.nn_model(x)
        # criterion
        target = train_batch[1]
        loss = torch.sqrt(self.loss(x, target))
        # logger
        metrics = {'loss': loss, }
        self.log_dict(metrics)
        return metrics

    def validation_step(self, val_batch, batch_idx):
        # prepare inputs
        x = val_batch[0]
        #target = torch.flatten(val_batch[1], start_dim=1)
        # process model
        x = self.nn_model(x)
        # criterion
        target = val_batch[1]
        loss = torch.sqrt(self.loss(x, target))
        # logger
        metrics = {'val_loss': loss, }
        print('val_loss', loss)
        self.log_dict(metrics)
        return metrics
