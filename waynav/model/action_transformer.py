from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertEmbeddings, BertPooler, BertPredictionHeadTransform
from .model import Rotation_Embeddings
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Img_Feature_Embedding(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, kernel_size=1,stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(128*100, output_size)
    def forward(self, visual_input):
        # B, Seq_len, Hidden, 10, 10
        shape = visual_input.size()
        visual_input = visual_input.view(-1, shape[2], shape[3], shape[4])
        # B*Seq_len, 128, 10, 10
        visual_feat = self.convs(visual_input)
        visual_feat = visual_feat.view(visual_feat.size(0), -1)
        visual_feat = self.fc(visual_feat)
        visual_feat = visual_feat.view(shape[0], shape[1], -1)

        return visual_feat

class VLN_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.image_embeddings = Img_Feature_Embedding(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.view_embeddings = Rotation_Embeddings(config)
        self.action_embeddings = nn.Embedding(48, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.post_init()

    def forward(self, input_ids, img_feat, act_seq, view_idx, attention_mask):
        view_idx_embeddings = self.view_embeddings(view_idx)
        resnet_embeddings = self.image_embeddings(img_feat)
        img_embeddings = self.dropout(resnet_embeddings+view_idx_embeddings)
        action_embeddings = self.action_embeddings(act_seq)

        word_embeddings = self.embeddings(input_ids['input_ids'], token_type_ids=input_ids['token_type_ids'])
        input_feats = torch.cat([word_embeddings, img_embeddings, action_embeddings], dim=1)
        
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(input_feats, attention_mask=extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs[0])

        return pooled_output

class VLN_Navigator(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = VLN_Encoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size*4, 9)
        self.direction_predictor = nn.Linear(config.hidden_size, 4)
        self.distance_predictor = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, img_feat, act_seq, view_idx, attention_mask):
        pooled_output = self.encoder(input_ids, img_feat, act_seq, view_idx, attention_mask)
        pooled_output = self.dropout(pooled_output)
        direction = self.direction_predictor(pooled_output)
        distance = self.distance_predictor(pooled_output)

        return direction, distance