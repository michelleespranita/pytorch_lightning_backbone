import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return nn.ReLU(self.l1(x.view(x.size(0), -1)))
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    