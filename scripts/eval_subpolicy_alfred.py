from waynav.eval import Eval_Subpolicy_Agent
import argparse
from transformers import AutoConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/ubuntu/alfred/data/json_2.1.0/valid_seen")
    parser.add_argument('--save_path', type=str, default="/home/ubuntu/data/output/test_waynav/valid_seen_gtsub")
    parser.add_argument('--metadata_path', type=str, default="/home/ubuntu/data/alfred_metadata/")
    parser.add_argument('--debug_dir', type=str, default="/home/ubuntu/data/output/test_waynav/valid_seen_gtsub")
    parser.add_argument('--object_model_path', type=str, default="/home/ubuntu/vln_detector/ckpt/newdata_object/model_final.pth")
    parser.add_argument('--recep_model_path', type=str, default="/home/ubuntu/vln_detector/ckpt/newdata_recep/model_final.pth")
    parser.add_argument('--detection_config_file', type=str, default="/home/ubuntu/vln_detector/configs/baseline_r50_2x.py")
    parser.add_argument('--subpolicy_model_path', type=str, default="/home/ubuntu/data/alfred_metadata/subpolicy/checkpoint-2000")
    # parser.add_argument('--ll_model_path', type=str, default="/home/ubuntu/data/alfred_metadata/ll_action/checkpoint-1000")
    parser.add_argument('--ll_model_path', type=str, default="/home/ubuntu/data/alfred_metadata/ll_action/last_checkpoint")
    parser.add_argument('--navigation_gpu', type=int, default=0)
    parser.add_argument('--detection_gpu', type=int, default=1)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--reward_config', type=str, default='/home/ubuntu/alfred/models/config/rewards.json')
    args = parser.parse_args()

    agent = Eval_Subpolicy_Agent(args)
    agent.run()