import torch
from waynav.data.pretrain_dataset import Pretrain_Dataset
from waynav.data.collate import Pretrain_Collator
from waynav.model.model import View_Selector, ROI_Waypoint_Predictor
from waynav.trainer.pretrain_trainer import PretrainTrainer
from transformers import TrainingArguments, AutoConfig
from torch.utils.data import DataLoader

eval_data_dir = '/data/joey/substep_train'
single_view = True
eval_dataset = Pretrain_Dataset(eval_data_dir, single_view=single_view)
tokenizer = eval_dataset.tokenizer
data_collator = Pretrain_Collator(tokenizer)

config = AutoConfig.from_pretrained('bert-base-uncased')
if single_view:
    model = ROI_Waypoint_Predictor(config).from_pretrained('/data/joey/waypoint/output/predictor/checkpoint-8500')
    output_path = "/data/joey/waypoint/output/predictor"
    batch_size = 32
    learning_rate = 5e-5
else:
    model = View_Selector(config)
    output_path = "/data/joey/waypoint/output/view_selector"
    batch_size = 8
    learning_rate = 1e-4

output_path = "/data/joey/waypoint/output/test"


training_args = TrainingArguments(
    output_dir=output_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    warmup_ratio=0.1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=20,
    weight_decay=0.0001,
    logging_steps=20,
    report_to='tensorboard'
)

trainer = PretrainTrainer(
    model=model,
    args=training_args,
    train_dataset=eval_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_predictor=single_view,
)

trainer.evaluate()

