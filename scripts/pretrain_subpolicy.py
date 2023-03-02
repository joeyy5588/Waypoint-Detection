import torch
from waynav.data import Subpolicy_Pretrain_Dataset
from waynav.data.collate import Subpolicy_Pretrain_Collator
from waynav.model import BartForSubpolicyPretrain, VLN_MetaAction
from waynav.trainer import SubpolicyPreTrainer
from transformers import TrainingArguments, AutoConfig
from torch.profiler import profile, record_function, ProfilerActivity
import yaml
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", '-c', default="configs/pretrain_subpolicy.yaml", metavar="FILE", help="path to config file")
parser.add_argument("--local_rank", type=int, default=0, help="the rank of this machine (unique per machine)")

args = parser.parse_args()
config = yaml.safe_load(open(args.config_file))
dataset_args = config['Dataset']
model_args = config['Model']
training_args = config['Training']
misc_arags = config['Misc']

train_dataset = Subpolicy_Pretrain_Dataset(dataset_args['train_data_dir'])
eval_dataset = Subpolicy_Pretrain_Dataset(dataset_args['eval_data_dir'])
unseen_eval_dataset = Subpolicy_Pretrain_Dataset(dataset_args['unseen_eval_data_dir'])
eval_dataset_dict = {
    'seen': eval_dataset,
    'unseen': unseen_eval_dataset
}
tokenizer = train_dataset.tokenizer
data_collator = Subpolicy_Pretrain_Collator(tokenizer)

config = AutoConfig.from_pretrained('bert-base-uncased')
config.update(model_args)

model = VLN_MetaAction.from_pretrained('bert-base-uncased', config=config)

trainer_args = TrainingArguments(
    **training_args
)

trainer = SubpolicyPreTrainer(
    model=model,
    args=trainer_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset_dict,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

