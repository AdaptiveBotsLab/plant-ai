# Plant AI
This project applies momentum contrast learning to the specialised field of plant microbiology. This is based off the He et al.'s Momentum Contrast (MoCo) for Unsupervised Visual Representation Learning (2020). Using MoCo, we build a self-supervised deep-learning model capable of understanding plant tissue structure, which can be adapted to more complicated tasks. Here, we compare the its performance against supervised ResNet18 models in three different species classification tasks.

# Prerequisites
Download the packages and libraries from the requirements.txt
`pip install -r requirements.txt`

# Quick Start
1. Download the dataset from Kaggle (https://www.kaggle.com/datasets/hxvoon/plant-tissue-cleaned-dataset) and place them in the same directory as the moco_plants_complete.py
2. Run the moco_plants_complete.py script
3. View results using `tensorboard --logdir tb_complete/logs`. You can change the output directory of specific loggers, e.g. 'moco_logger = desired_output_directory'

# Comparison with ResNet18
1. Run the vanilla resnet classifier. This will create three different ResNet18-based models checkpoints for each of the task.
2. Run any of the 'tl-resnet...' files to perform transfer learning from their saved checkpoints from step 1.
3. View results using `tensorboard --logdir tb_complete/logs`. Make sure to check the loggers you want to view.