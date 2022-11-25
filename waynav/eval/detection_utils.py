from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
# from detectron2.engine import default_setup
from detectron2.engine.defaults import create_ddp_model

def build_detection_model(args):
    cfg = LazyConfig.load(args.config_file)
    # 74 for object, 32 for receptacles
    cfg.model.roi_heads.num_classes = 74
    cfg.model.roi_heads.box_predictor.test_score_thresh = 0.2
    # default_setup(cfg, args)
    model = instantiate(cfg.model)
    model.eval()
    # model = create_ddp_model(model)
    DetectionCheckpointer(model).load(args.detection_model_path)

    return model