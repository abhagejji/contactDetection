
from torchvision import models
import torch.nn as nn
import torch
import pytorch_lightning as pl


class MyEfficientNet_3channel(pl.LightningModule):
    def __init__(self, num_classes, input_channels=3):
        super().__init__() # call the constructor of parents class
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(1280, self.num_classes)
        #self.train_acc = Accuracy()  # Create an instance of Accuracy metric for training

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # predicted_probs = torch.sigmoid(logits)
        # predicted_labels = (predicted_probs >= 0.5).float()  # Threshold at 0.5 for binary classification
        # accuracy = (predicted_labels.squeeze() == y).float().mean()  # Make sure to squeeze the predictions if needed

        # self.log('train_acc', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # predicted_probs = torch.sigmoid(logits)
        # predicted_labels = (predicted_probs >= 0.5).float()  # Threshold at 0.5 for binary classification
        # accuracy = (predicted_labels.squeeze() == y).float().mean()  # Make sure to squeeze the predictions if needed

        # self.log('train_acc', accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
