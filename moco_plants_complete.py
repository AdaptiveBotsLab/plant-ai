import copy

import os
import glob

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchmetrics

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models import ResNetGenerator
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
from lightly.transforms import MoCoV2Transform, utils

# Configuration
num_workers = 16
batch_size = 30
# memory bank = number of negatives kept in MoCo's dictionary. 
# When set > than total number of images, more identical negatives would be saved which does not improve performance. 
# Hence, maximum value should be total number of images
memory_bank_size = 4096
seed = 1
max_epochs_moco = 700
max_epochs_classifier = 400

# sets precision, highest = max accuracy slowest, medium = fastest weakest precision
torch.set_float32_matmul_precision("high")

## Paths

# Just use the species_part dataset
path_to_train = './plants_species_part/train'
#species_part
path_to_train_species_part = './plants_species_part/train/'
path_to_test_species_part = './plants_species_part/test/'
#stem_root
path_to_train_stem_root = './plants_stem_root/train/'
path_to_test_stem_root = './plants_stem_root/test/'
#dicot_monocot
path_to_train_dicot_monocot = './plants_dicot_monocot/train/'
path_to_test_dicot_monocot = './plants_dicot_monocot/test/'

pl.seed_everything(seed)

# creates a logger to log the information that can be displayed on tensorboard
moco_logger = TensorBoardLogger("tb_logs/complete", name="moco")
species_part_logger = TensorBoardLogger("tb_logs/complete", name="species_part")
stem_root_logger = TensorBoardLogger("tb_logs/complete", name="stem_root")
dicot_monocot_logger = TensorBoardLogger("tb_logs/complete", name="dicot_monocot")

# disable blur because we're working with tiny images
transform = MoCoV2Transform(
    input_size=224,
    gaussian_blur=0,
    random_gray_scale=0.2
)

train_classifier_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=utils.IMAGENET_NORMALIZE["mean"],
        std=utils.IMAGENET_NORMALIZE["std"],
    ),
])

test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=utils.IMAGENET_NORMALIZE["mean"],
        std=utils.IMAGENET_NORMALIZE["std"],
    ),
])

# We use the moco augmentations for training moco
dataset_train_moco = LightlyDataset(input_dir=path_to_train, transform=transform)

# lightlyfy datasets before dataloader
dataset_train_classifier_species_part = LightlyDataset(
    input_dir=path_to_train_species_part, transform=train_classifier_transforms
)
dataset_train_classifier_stem_root = LightlyDataset(
    input_dir=path_to_train_stem_root, transform=train_classifier_transforms
)
dataset_train_classifier_dicot_monocot = LightlyDataset(
    input_dir=path_to_train_dicot_monocot, transform=train_classifier_transforms
)


# test dataset
dataset_test_species_part = LightlyDataset(input_dir=path_to_test_species_part, transform=test_transforms)
dataset_test_stem_root = LightlyDataset(input_dir=path_to_test_stem_root, transform=test_transforms)
dataset_test_dicot_monocot = LightlyDataset(input_dir=path_to_test_dicot_monocot, transform=test_transforms)

# trainer dataloaders

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_train_classifier_species_part = torch.utils.data.DataLoader(
    dataset_train_classifier_species_part,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)
dataloader_train_classifier_stem_root = torch.utils.data.DataLoader(
    dataset_train_classifier_stem_root,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)
dataloader_train_classifier_dicot_monocot = torch.utils.data.DataLoader(
    dataset_train_classifier_dicot_monocot,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
)

dataloader_test_species_part = torch.utils.data.DataLoader(
    dataset_test_species_part,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
dataloader_test_stem_root = torch.utils.data.DataLoader(
    dataset_test_stem_root,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)
dataloader_test_dicot_monocot = torch.utils.data.DataLoader(
    dataset_test_dicot_monocot,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers,
)


# Creating the MoCo Lightning Module
class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = ResNetGenerator("resnet-18", width=1, num_splits=2) #num_splits simulate n number of GPU training
        #width is a very heavy parameter

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(memory_bank_size, 128)
        )

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        # get queries
        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        # get keys
        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        q_n = torch.nn.functional.normalize(q, dim=1)
        k_n = torch.nn.functional.normalize(k, dim=1)
    
        logits_preview = torch.matmul(q_n, k_n.T) 
        targets = torch.arange(len(q), device=q.device)

        _, preds = logits_preview.max(dim=1)
        acc = (preds == targets).float().mean()
        self.log("moco_training_accuracy", acc, on_epoch=True, prog_bar=True)
        

        loss = self.criterion(q, k)
        self.log("moco_ssl_nXent", loss, on_epoch=True) #nXentloss
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=0.01, # changed to see if loss lowers
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs_moco)
        return [optim], [scheduler]
    
# Create Classifier Lightning Module
class Classifier_species_part(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, 5)

        self.criterion = nn.CrossEntropyLoss()
                
        # Accuracy metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("species_classifier_training_accuracy", acc, on_epoch=True, prog_bar=True)

        self.log("species_classifier_training_ce_loss", loss, on_epoch=True)
        return loss

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        
        self.val_acc(preds, y)
        self.log("species_classifier_test_accuracy", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs_classifier)
        return [optim], [scheduler]
    
# probably the most inefficient way to do this
class Classifier_stem_root(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # freeze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, 2)

        self.criterion = nn.CrossEntropyLoss()
                
        # Accuracy metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("stem_root_classifier_training_accuracy", acc, on_epoch=True, prog_bar=True)

        self.log("stem_root_classifier_training_ce_loss", loss, on_epoch=True)
        return loss

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        
        self.val_acc(preds, y)
        self.log("stem_root_classifier_test_accuracy", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs_classifier)
        return [optim], [scheduler]
    
class Classifier_dicot_monocot(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone

        # free  ze the backbone
        deactivate_requires_grad(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(512, 2)

        self.criterion = nn.CrossEntropyLoss()
                
        # Accuracy metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        _, preds = torch.max(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log("dmcot_classifier_training_accuracy", acc, on_epoch=True, prog_bar=True)

        self.log("dmcot_classifier_training_ce_loss", loss, on_epoch=True)
        return loss

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        preds = torch.argmax(y_hat, dim=1)
        
        self.val_acc(preds, y)
        self.log("dmcot_classifier_test_accuracy", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs_classifier)
        return [optim], [scheduler]

if __name__ == "__main__":

    ## MoCo
    model = MocoModel()
    trainer_moco = pl.Trainer(
        max_epochs=max_epochs_moco,
        devices=1,
        accelerator="gpu",
        log_every_n_steps= 8,
        logger=moco_logger
    )
    
    trainer_moco.fit(model, train_dataloaders=dataloader_train_moco)

    ## Classifier Species+Part
    model.eval() # turns off dropout and batchnorm for actual testing
    classifier_species_part = Classifier_species_part(model.backbone)
    classifier_stem_root = Classifier_stem_root(model.backbone)
    classifier_dicot_monocot = Classifier_dicot_monocot(model.backbone)

    trainer_classifier_species_part = pl.Trainer(
        max_epochs=max_epochs_classifier,
        devices=1,
        accelerator="gpu",
        logger=species_part_logger
    )
    trainer_classifier_stem_root = pl.Trainer(
        max_epochs=max_epochs_classifier,
        devices=1,
        accelerator="gpu",
        logger=stem_root_logger
    )
    trainer_classifier_dicot_monocot = pl.Trainer(
        max_epochs=max_epochs_classifier,
        devices=1,
        accelerator="gpu",
        logger=dicot_monocot_logger
    )

    trainer_classifier_species_part.fit(
        classifier_species_part,
        train_dataloaders=dataloader_train_classifier_species_part,
        val_dataloaders=dataloader_test_species_part,
    )
    
    trainer_classifier_stem_root.fit(
        classifier_stem_root,
        train_dataloaders=dataloader_train_classifier_stem_root,
        val_dataloaders=dataloader_test_stem_root,
    )
    trainer_classifier_dicot_monocot.fit(
        classifier_dicot_monocot,
        train_dataloaders=dataloader_train_classifier_dicot_monocot,
        val_dataloaders=dataloader_test_dicot_monocot,
    )