from waynav.eval.agent import Eval_Agent
import argparse
from waynav.model.model import Waypoint_Transformer
from transformers import AutoConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/alfworld/data/json_2.1.1/valid_seen")
    parser.add_argument('--save_path', type=str, default="/mnt/alfworld/data/test_waynav")
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--reward_config', type=str, default='/mnt/alfworld/alfworld/agents/config/rewards.json')
    args = parser.parse_args()

    
    config = AutoConfig.from_pretrained('prajjwal1/bert-medium')
    model = Waypoint_Transformer(config, predict_xyz=True).from_pretrained('output/checkpoint-2500')
    model.eval()

    agent = Eval_Agent(args, model)
    agent.run()