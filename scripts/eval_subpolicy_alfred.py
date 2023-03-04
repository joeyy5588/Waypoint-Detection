from waynav.eval import Eval_Subpolicy_Agent
import argparse
from transformers import AutoConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/joey/alfworld/data/json_2.1.1/valid_seen")
    parser.add_argument('--save_path', type=str, default="/data/joey/output/test_waynav")
    parser.add_argument('--object_model_path', type=str, default="/home/joey/vln_detector/ckpt/newdata_object/model_final.pth")
    parser.add_argument('--recep_model_path', type=str, default="/home/joey/vln_detector/ckpt/newdata_recep/model_final.pth")
    parser.add_argument('--detection_config_file', type=str, default="/home/joey/vln_detector/configs/baseline_r50_2x.py")
    parser.add_argument('--subpolicy_model_path', type=str, default="/data/joey/alfred_metadata/subpolicy/checkpoint-2000")
    parser.add_argument('--ll_model_path', type=str, default="/data/joey/alfred_metadata/ll_action/checkpoint-2000")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--reward_config', type=str, default='/home/joey/alfworld/alfworld/agents/config/rewards.json')
    args = parser.parse_args()

    agent = Eval_Subpolicy_Agent(args)
    agent.run()