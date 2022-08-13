from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertEmbeddings, BertPooler
# from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
# from habitat_baselines.rl.ddppo.policy import resnet
from .resnet import RGB_ResNet, Depth_ResNet
import torch.nn as nn
import torch
import numpy as np

class Waypoint_Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # For regression
        # self.coord_classifier = nn.Linear(config.hidden_size, 2)
        # For discretized distance (classification)
        self.coord_classifier_x = nn.Linear(config.hidden_size, 65)
        self.coord_classifier_z = nn.Linear(config.hidden_size, 65)
        self.angle_classifier = nn.Linear(config.hidden_size, 13)
        self.rotation_classifier = nn.Linear(config.hidden_size, 4)

    def forward(self, pooled_output):
        pooled_output = self.dropout(pooled_output)
        coord_logits_x = self.coord_classifier_x(pooled_output)
        coord_logits_z = self.coord_classifier_x(pooled_output)
        angle_logits = self.angle_classifier(pooled_output)
        rotation_logits = self.rotation_classifier(pooled_output)
        return coord_logits_x, coord_logits_z, angle_logits, rotation_logits

class Panoramic_Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.coord_embedding = nn.Linear(2, config.hidden_size)
        self.angle_embedding = nn.Embedding(13, config.hidden_size)
        self.rotation_embedding = nn.Embedding(4, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_angle, input_rotation):
        # embeddings = self.coord_embedding(input_coord) + self.angle_embedding(input_angle) + self.rotation_embedding(input_rotation)
        embeddings = self.angle_embedding(input_angle) + self.rotation_embedding(input_rotation)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VlnResnetDepthEncoder(nn.Module):
    def __init__(
        self,
        observation_space,
        output_size=128,
        checkpoint="pretrained_models/gibson-2plus-resnet50.pth",
        backbone="resnet50",
        resnet_baseplanes=32,
        normalize_visual_inputs=False,
        trainable=False,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(trainable)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint)

            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue

                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v

            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            # self.visual_fc = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(
            #         np.prod(self.visual_encoder.output_shape), output_size
            #     ),
            #     nn.ReLU(True),
            # )
            None
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)


    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            # return self.visual_fc(x)
            return x

class Waypoint_Transformer(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.image_embeddings = Panoramic_Embeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.classifier = Waypoint_Predictor(config)
        self.merge_visual_mlp = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.post_init()

        self.rgb_resnet = RGB_ResNet(config)
        self.depth_resnet = Depth_ResNet(config)
        
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, input_ids, rgb_list, depth_list, panorama_angle, panorama_rotation):
        # input_rgb_feats = []
        # input_depth_feats = []
        # for i in range(len(rgb_list)):
        #     rgb_feat = self.rgb_resnet[rgb_list[i]]
        #     depth_feat = self.rgb_resnet[depth_list[i]]
        #     traj_feat = self.image_embeddings(input_coord[i], input_angle[i], input_rotation[i])
        #     input_rgb_feats.append(rgb_feat + traj_feat)
        #     input_depth_feats.append(depth_feat + traj_feat)

        # Dataparallel will split 
        batch_size = panorama_angle.size(0)
        # B * 12 * D -> (B*12) * D
        rgb_feats = self.rgb_resnet(rgb_list)
        depth_feats = self.depth_resnet(depth_list)
        traj_feat = self.image_embeddings(panorama_angle, panorama_rotation)

        word_embeddings = self.embeddings(input_ids['input_ids'])
        rgb_feats = rgb_feats.reshape(batch_size, 12, -1)
        depth_feats = depth_feats.reshape(batch_size, 12, -1)
        merged_visual_feats = self.merge_visual_mlp(torch.cat([rgb_feats, depth_feats], dim=-1))

        input_feats = torch.cat([word_embeddings, merged_visual_feats], dim=1)
        encoder_outputs = self.encoder(input_feats)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        coord_logits_x, coord_logits_z, angle_logits, rotation_logits = self.classifier(pooled_output)
        coord_logits_x = self.softmax(coord_logits_x)
        distance_bins = torch.tensor(np.linspace(-8,8,65), device=coord_logits_x.device)
        coord_logits_x = torch.sum(coord_logits_x * distance_bins, dim=-1)
        coord_logits_z = self.softmax(coord_logits_z)
        distance_bins = torch.tensor(np.linspace(-8,8,65), device=coord_logits_z.device)
        coord_logits_z = torch.sum(coord_logits_z * distance_bins, dim=-1)
        coord_logits = torch.stack([coord_logits_x, coord_logits_z],dim=1)
        

        return (coord_logits, angle_logits, rotation_logits)