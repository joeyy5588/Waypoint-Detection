import torch
from data.dataset import Panorama_Dataset
from data.collate import RGBD_Collator
from model.model import Waypoint_Transformer
from trainer.trainer import WaypointTrainer
from transformers import TrainingArguments, AutoConfig
from torch.utils.data import DataLoader

train_data_dir = '/mnt/alfworld/data/panorama_valid_seen'
eval_data_dir = '/mnt/alfworld/data/panorama_valid_seen'
predict_xyz = True
train_dataset = Panorama_Dataset(train_data_dir, predict_xyz=predict_xyz)
eval_dataset = Panorama_Dataset(eval_data_dir, predict_xyz=predict_xyz)
tokenizer = train_dataset.tokenizer
data_collator = RGBD_Collator(tokenizer)

config = AutoConfig.from_pretrained('bert-base-uncased')
model = Waypoint_Transformer(config, predict_xyz=predict_xyz).from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.0001,
    logging_steps=20,
    report_to='tensorboard'
)

trainer = WaypointTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    predict_xyz=predict_xyz,
)

trainer.train()

