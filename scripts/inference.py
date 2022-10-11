import torch
from waynav.data.dataset import Panorama_Dataset
from waynav.data.collate import RGBD_Collator
from waynav.model.model import Waypoint_Transformer
from waynav.trainer.trainer import WaypointTrainer
from transformers import TrainingArguments, AutoConfig
from torch.utils.data import DataLoader

eval_data_dir = '/datadrive_c/chengfu/waypoint/data/panorama_valid_seen'
predict_xyz = False
eval_dataset = Panorama_Dataset(eval_data_dir, predict_xyz=predict_xyz)
tokenizer = eval_dataset.tokenizer
data_collator = RGBD_Collator(tokenizer)

config = AutoConfig.from_pretrained('prajjwal1/bert-medium')
model = Waypoint_Transformer(config, predict_xyz=predict_xyz).from_pretrained('/datadrive_c/chengfu/waypoint/output/checkpoint-5500')

training_args = TrainingArguments(
    output_dir="/datadrive_c/chengfu/waypoint/inference",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    warmup_ratio=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.0001,
    logging_steps=20,
)

trainer = WaypointTrainer(
    model=model,
    args=training_args,
    train_dataset=eval_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    predict_xyz=predict_xyz,
)

trainer.evaluate()

