import torch
from data.dataset import Panorama_Dataset

train_data_dir = '/datadrive_c/chengfu/waypoint/data/panorama'
train_dataset = Panorama_Dataset(train_data_dir)