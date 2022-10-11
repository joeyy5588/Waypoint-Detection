import torch
from waynav.data.dataset import Panorama_Dataset, ROI_Dataset
from waynav.data.collate import RGBD_Collator, ROI_Collator
from waynav.model.model import Waypoint_Transformer, ROI_Navigator
from waynav.trainer.trainer import WaypointTrainer
from waynav.trainer.roi_trainer import WaypointROITrainer
from transformers import TrainingArguments, AutoConfig
from torch.utils.data import DataLoader

train_data_dir = '/data/joey/panorama_train'
eval_data_dir = '/data/joey/panorama_valid_seen'
predict_xyz = True
train_dataset = ROI_Dataset(train_data_dir, predict_xyz=predict_xyz)
eval_dataset = ROI_Dataset(eval_data_dir, predict_xyz=predict_xyz)
tokenizer = train_dataset.tokenizer
data_collator = ROI_Collator(tokenizer)

config = AutoConfig.from_pretrained('prajjwal1/bert-medium')
model = ROI_Navigator(config)

training_args = TrainingArguments(
    output_dir="/data/joey/waypoint/output/",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=20,
    weight_decay=0.0001,
    logging_steps=20,
    report_to='tensorboard'
)

trainer = WaypointROITrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    predict_xyz=predict_xyz,
)

trainer.train()

