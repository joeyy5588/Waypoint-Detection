import torch
from data.dataset import Panorama_Dataset
from data.collate import RGBD_Collator
from model.model import Waypoint_Transformer
from trainer.trainer import WaypointTrainer
from transformers import TrainingArguments, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

train_data_dir = '/mnt/alfworld/data/panorama_valid_seen'
train_dataset = Panorama_Dataset(train_data_dir)
tokenizer = train_dataset.tokenizer
data_collator = RGBD_Collator(tokenizer)
dataloader = DataLoader(train_dataset, collate_fn = data_collator)

max_value = []
for i, (inputs) in enumerate(tqdm(dataloader)):
    input_ids, rgb_list, depth_list, meta_dict = inputs
    target_coord = meta_dict['target_coord']
    max_value.append(torch.min(target_coord).item())

print(min(max_value))