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
path_to_train_dicot_monocot = './plants_dicot_monocot/train/'
path_to_test_dicot_monocot = './plants_dicot_monocot/test/'

# Loggers organized into an 'vanilla' subdirectory
parent_dir = "tb_logs/complete_2026_04_13/resnet18_vanilla"
species_logger = TensorBoardLogger(parent_dir, name="species_part_vanilla_resnet")
stem_logger = TensorBoardLogger(parent_dir, name="stem_root_vanilla_resnet")
dicot_logger = TensorBoardLogger(parent_dir, name="dicot_monocot_vanilla_resnet")

## Transforms (Standardized for all experiments)
train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=utils.IMAGENET_NORMALIZE["mean"], 
                 std=utils.IMAGENET_NORMALIZE["std"]),
])

test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=utils.IMAGENET_NORMALIZE["mean"], 
                 std=utils.IMAGENET_NORMALIZE["std"]),
])

def get_loaders(train_path, test_path):
    train_ds = LightlyDataset(input_dir=train_path, transform=train_transforms)
    test_ds = LightlyDataset(input_dir=test_path, transform=test_transforms)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

class UntrainedPlantClassifier(pl.LightningModule):
    def __init__(self, num_classes, label_prefix):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.label_prefix = label_prefix

    def forward(self, x):
        feat = self.backbone(x).flatten(start_dim=1)
        return self.fc(feat)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        
        # CHANGED: Added '_classifier_' and 'training_accuracy' to match MoCo script
        self.log(f"{self.label_prefix}_classifier_training_accuracy", acc, 
                 on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        acc = self.accuracy(y_hat, y)
        
        # CHANGED: Added '_classifier_' and 'test_accuracy' to match MoCo script
        self.log(f"{self.label_prefix}_classifier_test_accuracy", acc, 
                 on_epoch=True, prog_bar=True, batch_size=x.shape[0])

    def configure_optimizers(self):
        # We use a standard learning rate for training the full network
        return torch.optim.Adam(self.parameters(), lr=1e-4)

if __name__ == "__main__":
    # Define tasks
    task_configs = [
        {"path": (path_to_train_species_part, path_to_test_species_part), "classes": 5, "prefix": "species"},
        {"path": (path_to_train_stem_root, path_to_test_stem_root), "classes": 2, "prefix": "stem_root"},
        {"path": (path_to_train_dicot_monocot, path_to_test_dicot_monocot), "classes": 2, "prefix": "dmcot"} # Changed to 'dmcot' to match
    ]

    active_tasks = []

    # --- STAGE 1: 1-Epoch Sanity Check ---
    print("\n" + "!"*60)
    print("RUNNING 1-EPOCH SANITY CHECK FOR ALL MODELS")
    print("!"*60)

    for config in task_configs:
        train_loader, test_loader = get_loaders(config["path"][0], config["path"][1])
        model = UntrainedPlantClassifier(num_classes=config["classes"], label_prefix=config["prefix"])
        
        # Simple trainer: No logging, no checkpoints, just 1 epoch
        scout_trainer = pl.Trainer(
            max_epochs=1,
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False # Keeps terminal cleaner
        )
        
        print(f"\n[Scouting] Task: {config['prefix']}")
        scout_trainer.fit(model, train_loader, test_loader)
        
        active_tasks.append({
            "model": model, 
            "loaders": (train_loader, test_loader)
        })

    # --- STAGE 2: Full Training ---
    loggers = [species_logger, stem_logger, dicot_logger]
    
    for task_info, task_logger in zip(active_tasks, loggers):
        model = task_info["model"]
        train_loader, test_loader = task_info["loaders"]
        
        trainer = pl.Trainer(
            max_epochs=max_epochs_classifier,
            accelerator="gpu",
            devices=1,
            logger=task_logger, # This connects the specific logger
            enable_checkpointing=True,
            log_every_n_steps=5 # Ensures logs appear even with small datasets
        )
        
        print(f"\n Training {model.label_prefix}...")
        trainer.fit(model, train_loader, test_loader)