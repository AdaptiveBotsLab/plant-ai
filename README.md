# Plant AI
This project applies momentum contrast learning to the specialised field of plant microbiology.

# Prerequisites
Download the packages and libraries from the requirements.txt
`pip install -r requirements.txt`

# Quick Start
1. Download the dataset from Kaggle (https://www.kaggle.com/datasets/hxvoon/plant-tissue-cleaned-dataset) and place them in the same directory as the moco_plants_complete.py
2. Run the moco_plants_complete.py script
3. View results using `tensorboard --logdir tb_complete/logs`. You can change the output directory of specific loggers, e.g. 'moco_logger = desired_output_directory'
