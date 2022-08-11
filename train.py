import torch
from data import Panorama_Dataset

train_data_dir = '/datadrive_c/chengfu/waypoint/data/panorama'
train_dataset = Panorama_Dataset(train_data_dir)