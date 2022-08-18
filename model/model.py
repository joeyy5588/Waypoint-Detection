from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertEmbeddings, BertPooler, BertPredictionHeadTransform
# from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
# from habitat_baselines.rl.ddppo.policy import resnet
from .resnet import RGB_ResNet, Depth_ResNet
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Waypoint_Predictor(nn.Module):
    def __init__(self, config, coord1_classes, coord2_classes):
        super().__init__()
        # BertPooler
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

        # Classification head
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.angle_classifier = nn.Linear(config.hidden_size, 13)
        self.rotation_classifier = nn.Linear(config.hidden_size, 4)

        # For regression
        # self.coord_classifier = nn.Linear(config.hidden_size, 2)
        # For discretized distance (classification)
        '''
            For xyz: 1 = x, 2 = z
            For polar: 1 = r, 2 = a+bi
        '''
        self.transform = nn.Linear(config.hidden_size, 128)
        self.relu = nn.ReLU()
        self.imaginary_plane = nn.Linear(128, 2)
        self.radius = nn.Linear(128, 1)
        

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.tanh(pooled_output)
        pooled_output = self.dropout(pooled_output)
        angle_logits = self.angle_classifier(pooled_output)
        rotation_logits = self.rotation_classifier(pooled_output)

        pooled_output_2 = self.transform(first_token_tensor)
        pooled_output_2 = self.relu(pooled_output_2)
        img_number = self.imaginary_plane(pooled_output_2)
        img_number = F.normalize(img_number)
        radius = self.radius(pooled_output_2)
        coordinate = img_number * radius


        return coordinate, angle_logits, rotation_logits

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
    def __init__(self, config, predict_xyz=True):
        super().__init__(config)
        self.config = config
        self.predict_xyz = predict_xyz

        '''
            For xyz: 1 = x, 2 = z range: 0 ~ 16
            For polar: 1 = r, 2 = theta range: 0 ~ 10
        '''
        if predict_xyz:
            self.coord1_classes = 65
            self.coord2_classes = 65
        else:
            self.coord1_classes = 51
            self.coord2_classes = 121

        self.embeddings = BertEmbeddings(config)
        self.image_embeddings = Panoramic_Embeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.classifier = Waypoint_Predictor(config, self.coord1_classes, self.coord2_classes)
        self.merge_visual_mlp = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.post_init()

        self.rgb_resnet = RGB_ResNet(config)
        self.depth_resnet = Depth_ResNet(config)
        if predict_xyz:
            self.coord1_space = np.linspace(0, 16, self.coord1_classes)
            self.coord2_space = np.linspace(0, 16, self.coord2_classes)
        else:
            self.coord1_space = np.linspace(0, 10, self.coord1_classes)
            self.coord2_space = np.linspace(0, 360, self.coord2_classes)

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
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if self.predict_xyz:
            coord_logits, angle_logits, rotation_logits = self.classifier(sequence_output)
        else:
            coord_logits_1, coord_logits_2, angle_logits, rotation_logits = self.classifier(sequence_output)

            coord_logits_1 = self.softmax(coord_logits_1)
            bin_values1 = torch.tensor(self.coord1_space, device=coord_logits_1.device)
            coord_logits_1 = torch.sum(coord_logits_1 * bin_values1, dim=-1)

            coord_logits_2 = self.softmax(coord_logits_2)
            bin_values2 = torch.tensor(self.coord2_space, device=coord_logits_2.device)
            coord_logits_2 = torch.sum(coord_logits_2 * bin_values2, dim=-1)

            coord_logits = torch.stack([coord_logits_1, coord_logits_2],dim=1)
        

        return (coord_logits, angle_logits, rotation_logits)

    def softmax(self, input, t=0.1):
        nom = torch.exp(input/t)
        denom = torch.sum(nom, dim=1).unsqueeze(1)
        return nom / denom

