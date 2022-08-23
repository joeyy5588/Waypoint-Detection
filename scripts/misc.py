import torch
from data.dataset import Panorama_Dataset
from data.collate import RGBD_Collator
from model.model import Waypoint_Transformer
from trainer.trainer import WaypointTrainer
from transformers import TrainingArguments, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import azimuthAngle
import math

train_data_dir = '/mnt/alfworld/data/panorama'
train_dataset = Panorama_Dataset(train_data_dir)
tokenizer = train_dataset.tokenizer
data_collator = RGBD_Collator(tokenizer)
dataloader = DataLoader(train_dataset, collate_fn = data_collator)

traj_list = train_dataset.traj_list

length = set()
angle = set()
d_x = set()
d_z = set()
for traj_data in tqdm(traj_list):
    x = traj_data['traj']['x']
    z = traj_data['traj']['z']

    for i in range(len(x)-1):
        delta_x = x[i+1] - x[i]
        delta_z = z[i+1] - z[i]

        d_x.add(delta_x)
        d_z.add(delta_z)

        l = math.sqrt(delta_x ** 2 + delta_z ** 2)
        length.add(l)

        angle.add(azimuthAngle(0,0,delta_x, delta_z))

print(len(length), max(length), min(length))
print(len(angle), max(angle), min(angle))
print(len(d_x), max(d_x), min(d_x))
print(len(d_z), max(d_z), min(d_z))