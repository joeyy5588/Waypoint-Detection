import torch
from data.dataset import Panorama_Dataset
from transformers import TrainingArguments

train_data_dir = '/mnt/alfworld/data/panorama'
train_dataset = Panorama_Dataset(train_data_dir)
train_dataset.__getitem__(0)