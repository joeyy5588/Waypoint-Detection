from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import detectron2.data.transforms as T
# from detectron2.engine import default_setup
from detectron2.engine.defaults import create_ddp_model
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import torch
import numpy as np
from waynav.gen import constants

def get_object_classes(object_type):

    OBJECTS_DETECTOR = constants.OBJECTS_DETECTOR
    STATIC_RECEPTACLES = constants.STATIC_RECEPTACLES
    ALL_DETECTOR = constants.ALL_DETECTOR
    
    if object_type == "objects":
        return OBJECTS_DETECTOR
    elif object_type == "receptacles":
        return STATIC_RECEPTACLES
    else:
        return ALL_DETECTOR

class Detection_Helper:
    def __init__(self, args, model_path, object_types='objects'):
        self.object_classes = get_object_classes(object_types)
        num_classes = len(self.object_classes)
        self.object_classes = np.array(self.object_classes)

        cfg = LazyConfig.load(args.detection_config_file)
        cfg.model.roi_heads.num_classes = num_classes
        cfg.model.roi_heads.box_predictor.test_score_thresh = 0.2
        self.model = instantiate(cfg.model)
        self.model.eval()

        self.aug = T.ResizeShortestEdge(
            [400, 400], 1333
        )

        DetectionCheckpointer(self.model).load(model_path)

        # self.idx2class = {i+1: self.object_classes[i] for i in range(len(self.object_classes))}

    def to_device(self, device):
        self.model.to(device)

    def preprocess_transform(self, rgb_list):
        batched_inputs = []
        for x in rgb_list:
            single_input = {
                'file_name': 'image.png',
                'image_id': 0,
                'height': 300,
                'width': 300,
                'image': torch.as_tensor(np.ascontiguousarray(x.transpose(2, 0, 1)))
            }
            batched_inputs.append(single_input)

        return batched_inputs

    def extract_cnn_features(self, batched_inputs):
        batched_inputs = self.preprocess_transform(batched_inputs)
        images = self.model.preprocess_image(batched_inputs)
        features = self.model.backbone.bottom_up(images.tensor)

        return features['res5']

    def extract_roi_features(self, batched_inputs):
        batched_inputs = self.preprocess_transform(batched_inputs)
        images = self.model.preprocess_image(batched_inputs)
        features = self.model.backbone(images.tensor)
        

        if self.model.proposal_generator is not None:
            proposals, _ = self.model.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.model.device) for x in batched_inputs]

        results, box_features = self.roi_forward(images, features, proposals)
        

        assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
        results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        pred_classes = []
        pred_boxes = []
        for i in range(len(results)):
            prediction = results[i]['instances']
            pred_class = self.object_classes[prediction.pred_classes.cpu().numpy()]
            pred_box = prediction.pred_boxes.tensor.cpu().numpy()
            pred_classes.append(pred_class)
            pred_boxes.append(pred_box)

        return pred_classes, pred_boxes, box_features

    def roi_forward(self, images, features, proposals):
        features = [features[f] for f in self.model.roi_heads.box_in_features]
        box_features = self.model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.model.roi_heads.box_head(box_features)
        predictions = self.model.roi_heads.box_predictor(box_features)

        pred_instances, pred_ind = self.model.roi_heads.box_predictor.inference(predictions, proposals)
        extracted_features = []
        for i in range(len(pred_ind)):
            extracted_features.append(box_features[pred_ind[i]])
        return pred_instances, extracted_features

    def predict_mask(self, original_image):
        with torch.no_grad():
            original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            outputs = self.model([inputs])[0]['instances']
            prediction_dict = {
                'pred_classes': self.object_classes[outputs.pred_classes.cpu().numpy()],
                'scores': outputs.scores.cpu().numpy(),
                'pred_masks': outputs.pred_masks.cpu().numpy()
            }
            return prediction_dict