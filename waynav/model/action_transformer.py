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

class ActSeq_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.action_embeddings = nn.Embedding(7, config.hidden_size)
        self.distance_embeddings = nn.Embedding(41, config.hidden_size)
        self.step_embeddings = nn.Embedding(15, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, act_seq, act_dist, act_step):
        act_embed = self.action_embeddings(act_seq)
        dist_embed = self.distance_embeddings(act_dist)
        step_embed = self.step_embeddings(act_step)
        embeddings = act_embed + dist_embed + step_embed
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VLN_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.image_embeddings = Img_Feature_Embedding(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.view_embeddings = Rotation_Embeddings(config)
        self.actseq_embeddings = ActSeq_Embedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.post_init()

    def forward(self, input_ids, obj_input_ids, img_feat, act_seq, act_dist, \
                act_step, view_idx, view_step, attention_mask):
        view_idx_embeddings = self.view_embeddings(view_idx)
        resnet_embeddings = self.image_embeddings(img_feat)
        img_embeddings = resnet_embeddings + view_idx_embeddings
        # img_embeddings = self.dropout(img_embeddings)
        action_embeddings = self.actseq_embeddings(act_seq, act_dist, act_step)

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

class VLNEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)


        self.image_embeddings = Img_Feature_Embedding(config.hidden_size)
        self.view_idx_embeddings = nn.Embedding(5, config.hidden_size)
        self.view_step_embeddings = nn.Embedding(3, config.hidden_size)

        self.distance_embeddings = nn.Embedding(41, config.hidden_size)
        self.action_step_embeddings = nn.Embedding(15, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, input_ids, obj_input_ids, img_feat, act_seq, act_dist, act_step, view_idx, view_step):
        
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        position_ids = self.position_ids[:, 0 : seq_length]

        if hasattr(self, "token_type_ids"):
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        instruction_embeddings = self.LayerNorm(embeddings)
        instruction_embeddings = self.dropout(instruction_embeddings)

        obj_word_embeddings = self.word_embeddings(obj_input_ids)
        img_feat_embeddings = self.image_embeddings(img_feat)
        view_step_embeddings = self.view_step_embeddings(view_step)
        view_idx_embeddings = self.view_idx_embeddings(view_idx)
        visual_embeddings = torch.cat([img_feat_embeddings, obj_word_embeddings], dim=1)
        visual_embeddings = visual_embeddings + view_step_embeddings + view_idx_embeddings
        visual_embeddings = self.LayerNorm(visual_embeddings)
        visual_embeddings = self.dropout(visual_embeddings)

        act_embed = self.word_embeddings(act_seq)
        dist_embed = self.distance_embeddings(act_dist)
        step_embed = self.action_step_embeddings(act_step)
        action_embeddings = act_embed + dist_embed + step_embed
        action_embeddings = self.LayerNorm(action_embeddings)
        action_embeddings = self.dropout(action_embeddings)

        # print(instruction_embeddings.size(), visual_embeddings.size(), action_embeddings.size())

        input_feats = torch.cat([instruction_embeddings, visual_embeddings, action_embeddings], dim=1)
        # input_feats = torch.cat([instruction_embeddings, visual_embeddings], dim=1)

        return input_feats

class VLN_Navigator(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = VLNEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size*4, 9)
        # self.direction_predictor = nn.Linear(config.hidden_size, 4)
        self.distance_predictor = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, obj_input_ids, img_feat, act_seq, act_dist, \
                act_step, view_idx, view_step, attention_mask):

        input_feats = self.embeddings(input_ids, obj_input_ids, img_feat, act_seq, act_dist, act_step, view_idx, view_step)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(input_feats, attention_mask=extended_attention_mask)
        pooled_output = self.pooler(encoder_outputs[0])
        pooled_output = self.dropout(pooled_output)
        # direction = self.direction_predictor(pooled_output)
        distance = self.distance_predictor(pooled_output)
        return 0, distance
        # return direction, distance