import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2
import torchmetrics
from lightly.data import LightlyDataset
from lightly.transforms import utils
from pytorch_lightning.loggers import TensorBoardLogger

# Configuration
num_workers = 16
batch_size = 30
seed = 1
max_epochs_classifier = 400
torch.set_float32_matmul_precision("high")
pl.seed_everything(seed)

## Paths
path_to_train_species_part = './plants_species_part/train/'
path_to_test_species_part = './plants_species_part/test/'
path_to_train_stem_root = './plants_stem_root/train/'
path_to_test_stem_root = './plants_stem_root/test/'

# Using the Dicot/Monocot checkpoint as the source
ckpt_path = './tl_checkpoints/tl_dmcot_epoch=399-step=5200.ckpt' 

parent_dir = "tb_logs/complete_2026_04_13_lr0_01/resnet18_vanilla"
species_logger = TensorBoardLogger(parent_dir, name="tl_dmcot_for_species_part")
stem_logger = TensorBoardLogger(parent_dir, name="tl_dmcot_for_stem_root")

## Transforms
train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=utils.IMAGENET_NORMALIZE["mean"], std=utils.IMAGENET_NORMALIZE["std"]),
])

test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=utils.IMAGENET_NORMALIZE["mean"], std=utils.IMAGENET_NORMALIZE["std"]),
])

def get_loaders(train_path, test_path):
    train_ds = LightlyDataset(input_dir=train_path, transform=train_transforms)
    test_ds = LightlyDataset(input_dir=test_path, transform=test_transforms)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

class TransferClassifier(pl.LightningModule):
    def __init__(self, num_classes, task_name, checkpoint_path=None):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt['state_dict']
            new_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
            self.backbone.load_state_dict(new_state_dict)
            
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(512, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.task_name = task_name 

    def forward(self, x):
        feat = self.backbone(x).flatten(start_dim=1)
        return self.fc(feat)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        if self.task_name == "species_part":
            tag = "species_classifier_training_accuracy"
        elif self.task_name == "stem_root":
            tag = "stem_root_classifier_training_accuracy"
        else: # dicot_monocot
            tag = "dmcot_classifier_training_accuracy"

        self.log(tag, self.accuracy(y_hat, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        
        if self.task_name == "species_part":
            tag = "species_classifier_test_accuracy"
        elif self.task_name == "stem_root":
            tag = "stem_root_classifier_test_accuracy"
        else: # dicot_monocot
            tag = "dmcot_classifier_test_accuracy"

        self.log(tag, self.accuracy(y_hat, y), on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        
    def configure_optimizers(self):
    # We use a standard learning rate for training the full network
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    # Species+part
    species_logger = TensorBoardLogger(parent_dir, name="tl_dmcot_for_species_part")
    train_sp, test_sp = get_loaders(path_to_train_species_part, path_to_test_species_part)
    model_sp = TransferClassifier(num_classes=5, task_name="species_part", checkpoint_path=ckpt_path)
    
    trainer_sp = pl.Trainer(
        max_epochs=max_epochs_classifier, 
        accelerator="gpu", 
        devices=1, 
        logger=species_logger, 
        log_every_n_steps=5
    )
    trainer_sp.fit(model_sp, train_sp, test_sp)

    #Stem roots
    stem_logger = TensorBoardLogger(parent_dir, name="tl_dmcot_for_stem_root")
    train_sr, test_sr = get_loaders(path_to_train_stem_root, path_to_test_stem_root)
    model_sr = TransferClassifier(num_classes=2, task_name="stem_root", checkpoint_path=ckpt_path)
    
    trainer_sr = pl.Trainer(
        max_epochs=max_epochs_classifier, 
        accelerator="gpu", 
        devices=1, 
        logger=stem_logger, 
        log_every_n_steps=5
    )
    trainer_sr.fit(model_sr, train_sr, test_sr)