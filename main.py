
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, CosineAnnealingLR, ChainedScheduler
from torch.optim import AdamW
import pytorch_lightning as pl

import torchmetrics
import torch.nn as nn
from einops import rearrange

from pytorch_lightning.callbacks import EarlyStopping, Callback, ModelCheckpoint
from model import UNet

from data import LandCoverDataset
import segmentation_models_pytorch as smp


class DataModule(pl.LightningDataModule):
    def __init__(self, train_data_dir):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.batch_size = 32

    def setup(self, stage=None):
        # Initialize the dataset and split it into train and validation sets
        full_dataset = LandCoverDataset(self.train_data_dir, transform=True, img_size=(256, 256))

        # Calculate sizes for train and validation splits
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        # Split the dataset
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,num_workers=4,persistent_workers=True,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers=4,persistent_workers=True)
class LandCoverModel(pl.LightningModule):
    def __init__(self, num_classes=7, learning_rate=1e-3, weight_decay=1e-5,dropout=0.2):
        super().__init__()
        self.save_hyperparameters()

        # Model
       # self.model = UNet(in_channels=3, num_classes=num_classes,dropout_p=dropout)
        self.model = smp.Unet(encoder_name="resnet34",encoder_weights="imagenet",in_channels=3,classes=7)
        self.learning_rate = learning_rate

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes)
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler with warmup and cosine annealing
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=5  # 5 epochs of warmup
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - 5,  # Remaining epochs after warmup
            eta_min=1e-8
        )

        # Chain the schedulers
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler
            },
        }
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)
        self.train_iou.update(preds, masks)
        self.train_accuracy.update(preds, masks)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.criterion(outputs, masks)

        preds = torch.argmax(outputs, dim=1)
        self.val_iou.update(preds, masks)
        self.val_accuracy.update(preds, masks)

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', self.val_accuracy, on_step=False, on_epoch=True)

        return loss



if __name__ == "__main__":
    from torch.multiprocessing import freeze_support

    train_data_dir = ''

    torch.set_float32_matmul_precision('medium')
    freeze_support()

    # Initialize model and data module
    model = LandCoverModel(num_classes=7,learning_rate=3e-4)
    data_module = DataModule(
        train_data_dir=train_data_dir,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="pre-trained/",
        filename="landcover-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        callbacks=[checkpoint_callback, early_stopping],
        check_val_every_n_epoch=1,
        log_every_n_steps=5,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)