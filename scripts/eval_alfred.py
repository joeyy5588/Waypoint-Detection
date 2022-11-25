from waynav.eval.agent import Eval_Agent
import argparse
from waynav.model.model import Waypoint_Transformer
from transformers import AutoConfig
from 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/data/joey/alfworld/data/json_2.1.1/valid_seen")
    parser.add_argument('--save_path', type=str, default="/data/joey/output/test_waynav")
    parser.add_argument('--detection_model_path', type=str, default="/home/joey/vln_detector/ckpt/newdata_object")
    parser.add_argument('--direction_model_path', type=str, default="")
    parser.add_argument('--distance_model_path', type=str, default="")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--reward_config', type=str, default='/data/joey/alfworld/alfworld/agents/config/rewards.json')
    args = parser.parse_args()

    agent = Eval_Agent(args)
    agent.run()