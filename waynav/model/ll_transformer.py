from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertEmbeddings, BertPooler, BertPredictionHeadTransform
import torch.nn as nn
import torch.nn.functional as F
import torch

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


class VLNEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)


        self.image_embeddings = Img_Feature_Embedding(config.hidden_size)
        self.view_idx_embeddings = nn.Embedding(5, config.hidden_size)
        self.view_step_embeddings = nn.Embedding(3, config.hidden_size)
        self.subpolicy_embeddings = nn.Embedding(9, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(self, input_ids, obj_input_ids, img_feat, view_idx, subpolicy, view_step):
        
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
        view_idx_embeddings = self.view_idx_embeddings(view_idx)
        view_step_embeddings = self.view_step_embeddings(view_step)
        visual_embeddings = torch.cat([img_feat_embeddings, obj_word_embeddings], dim=1)
        visual_embeddings = visual_embeddings + view_idx_embeddings + view_step_embeddings
        visual_embeddings = self.LayerNorm(visual_embeddings)
        visual_embeddings = self.dropout(visual_embeddings)

        subpolicy_embeddings = self.subpolicy_embeddings(subpolicy)
        subpolicy_embeddings = subpolicy_embeddings.unsqueeze(1)

        input_feats = torch.cat([instruction_embeddings, subpolicy_embeddings, visual_embeddings], dim=1)
        # input_feats = torch.cat([instruction_embeddings, visual_embeddings], dim=1)

        return input_feats


class VLN_LL_Action(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = VLNEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.action_predictor = nn.Linear(config.hidden_size, 4)
        # self.boundary_predictor = nn.Linear(config.hidden_size, 2)
            
        self.post_init()

    def forward(self, input_ids, obj_input_ids, img_feat, view_idx, subpolicy, view_step, attention_mask, labels=None):

        input_feats = self.embeddings(input_ids, obj_input_ids, img_feat, view_idx, subpolicy, view_step)

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
        action = self.action_predictor(pooled_output)
        #boundary = self.boundary_predictor(pooled_output)
        return action#, boundary