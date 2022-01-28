import torch
from pytorch_lightning import LightningModule
import timm
from torch.optim import Adam
import torch.nn as nn
import numpy as np

from utils.constant import ATTRIBUTES

class Classification(LightningModule):

    def __init__(self, config, attr_dict=None):
        """method used to define our model parameters"""
        super().__init__()

        #self.loss = nn.BCELoss()
        self.loss = nn.BCEWithLogitsLoss()
        self.attr_dict = attr_dict

        # optimizer parameters
        self.lr = config.lr if hasattr(config, 'lr') else None

        self.net = timm.create_model(config.model_name, 
                                    pretrained=config.pretrained, 
                                    num_classes=config.n_classes)

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train/loss", loss)

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val/loss", loss)

        # Let's return preds to use it in a custom callback
        return {"logits": logits}

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("test/loss", loss)

        return {"logits": logits}

    def predict_step(self, batch, batch_idx):
        x, img_name = batch


        logits = self(x)
        converted_logits = nn.Sigmoid()(logits.detach())
        preds = torch.round(converted_logits)

        converted_preds = preds.detach().cpu().numpy()
        batch_converted_preds = []
        for pred_batch in converted_preds:
            batch_converted_preds.append([ATTRIBUTES[i] for i in np.where(pred_batch==1.0)[0]])
        return img_name, preds, batch_converted_preds, converted_logits.cpu().numpy()

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        return optimizer

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y.float())
        return loss, logits.detach()