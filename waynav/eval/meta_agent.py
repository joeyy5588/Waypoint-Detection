import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from waynav.gen.utils.py_util import walklevel
from waynav.env.thor_env import ThorEnv
from transformers import AutoTokenizer, AutoConfig
from .detection_utils import Detection_Helper
from waynav.model import VLN_Navigator, ROI_Waypoint_Predictor
from .navigation_utils import prepare_direction_input, prepare_distance_input, predict_direction, predict_distance
import math
import torch

class Eval_Agent(object):
    def __init__(self, args):
        # Path for testing data
        data_path = args.data_path
        save_path = args.save_path
        split = data_path.split('/')[-1]
        self.args = args
        self.use_gt_nav = True
        self.use_gt_mask = True
        self.use_gt_subpolicy = True
        self.traj_list = []
        # self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.object_model = Detection_Helper(args, args.object_model_path)
        self.recep_model = Detection_Helper(args, args.recep_model_path, object_types='receptacles')

        # cache
        cache_file = os.path.join(args.save_path, "cache.json")
        if os.path.isfile(cache_file):
            with open(cache_file, 'r') as f:
                finished_jsons = json.load(f)
        else:
            finished_jsons = {'finished': []}

        self.finished_list = finished_jsons['finished']

        for dir_name, subdir_list, file_list in walklevel(data_path, level=2):
            if "trial_" in dir_name:
                json_file = os.path.join(dir_name, 'traj_data.json')
                self.traj_list.append(json_file)


    def run(self):
        env = ThorEnv(player_screen_width=300, player_screen_height=300)
        traj_list = self.traj_list
        finished_list = self.finished_list

        trial_num = 0
        trial_success = 0
        goal_num = 0
        goal_success = 0
        while len(traj_list) > 0:
            json_file = traj_list.pop()

            print ("(%d Left) Evaluating: %s" % (len(traj_list), json_file))
            try:
                success, gc, gn = self.execute_navigation(self, env, json_file)
                trial_num += 1
                trial_success += success
                goal_num += gc
                goal_success += gn

                print("Complete %d/%d tasks, and %d/%d goals."%(trial_success, trial_num, goal_success, goal_num))
                finished_list.append(json_file)
                

            except Exception as e:
                import traceback
                traceback.print_exc()
                print ("Error: " + repr(e))

        env.stop()
        print("Finished.")
        print("Task SR: %f, Task GC: %f" %(trial_success/ trial_num, goal_success/goal_num))

    @classmethod
    def plot_waypoint(cls, waypoint_file, predict_waypoint):
        data = json.load(open(waypoint_file))
        navpoint = data['navigation_point']
        fig = plt.figure()
        predict_x = [x[0] for x in predict_waypoint]
        predict_z = [x[1] for x in predict_waypoint]

        plt.scatter(data['traj']['x'], data['traj']['z'])
        plt.scatter(data['traj']['x'][navpoint[1]], data['traj']['z'][navpoint[1]])
        plt.scatter(data['traj']['x'][0], data['traj']['z'][0])
        plt.scatter(predict_x[0], predict_z[0])
        try:
            plt.scatter(predict_x[1], predict_z[1])
        except:
            pass

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite('Traj.png', img)

    def execute_navigation(self, env, traj_data):
        traj_data = json.load(open(json_file))
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        # reset
        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        event = env.step(dict(traj_data['scene']['init_action']))

        env.set_task(traj_data, self.args, reward_type='dense')

        if self.use_gt_nav:
            for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
                cmd = ll_action['api_action']
                cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if k in cmd}
                event = env.step(cmd)
                if not event.metadata['lastActionSuccess']:
                    raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))

        if env.get_goal_satisfied():
            print("Goal Reached")
            success = 1
        else:
            print("Goal Not Reached")
            success = 0

        goal_success, goal_num = env.get_goal_conditions_met()

        return success, goal_success, goal_num



    def execute_interaction(self, env):
        pass

    def predict_waypoint(self, env, json_file, waypoint_file):
        device='cuda:0'
        self.object_model.to_device(device)
        self.recep_model.to_device(device)
        self.distance_model.to(device)
        self.distance_model.eval()

        self.direction_model.to('cuda:2')
        self.direction_model.eval()

        traj_data = json.load(open(json_file))
        waypoint_data = json.load(open(waypoint_file))
        traj_data['images'] = list()

        navpoint_idx = waypoint_data['navigation_point']

        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        object_toggles = traj_data['scene']['object_toggles']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']

        # reset
        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        event = env.step(dict(traj_data['scene']['init_action']))

        env.set_task(traj_data, self.args, reward_type='dense')

        fail = 0
        progress_indicator = 0
        total_goals = len(navpoint_idx)
        complete_goals = 0
        predicted_waypoints = []
        actseq_list = [1]
        dist_list = [0]
        while fail < 14 and complete_goals < total_goals:
            last_event = event
            current_nav_point = \
            (waypoint_data['traj']['x'][navpoint_idx[progress_indicator]+1], waypoint_data['traj']['z'][navpoint_idx[progress_indicator]+1])
            current_instruction = waypoint_data['instructions'][navpoint_idx[progress_indicator]]

            rgb_image, depth_image = self.get_panorama_image(env, event)
            with torch.no_grad():
                patch_feat = self.object_model.extract_cnn_features(rgb_image)
                obj_cls, obj_box, obj_feat = self.object_model.extract_roi_features(rgb_image)
                recep_cls, recep_box, recep_feat = self.recep_model.extract_roi_features(rgb_image)

            direction_input = prepare_direction_input(self.tokenizer, patch_feat, obj_cls, recep_cls, current_instruction, actseq_list, dist_list)
            pred_dir, pred_dist = predict_direction(self.direction_model, direction_input)
            if pred_dir == 0:
                event = env.step({'action': 'RotateLeft', 'forceAction': True})
            elif pred_dir == 2:
                event = env.step({'action': 'RotateRight', 'forceAction': True})
            elif pred_dir == 3:
                event = env.step({'action': 'RotateRight', 'forceAction': True})
                event = env.step({'action': 'RotateRight', 'forceAction': True})

            # distance_input = prepare_distance_input(self.tokenizer, obj_feat, obj_box, obj_cls, recep_feat, recep_box, recep_cls, current_instruction, pred_dir)
            # pred_dist = predict_distance(self.distance_model, distance_input)

            move_success = 0
            for i in range(pred_dist):
                event = env.step({'action': 'MoveAhead', 'forceAction': True})
                if event.metadata['lastActionSuccess']:
                    move_success += 1
                else:
                    break
                    # print(event.metadata['errorMessage'])

            actseq_list.append(pred_dir)
            dist_list.append(move_success)

            next_action = event.metadata["agent"]['position']
            # print(last_event.metadata["agent"]['position'], current_nav_point, next_action, pred_dir, pred_dist)
            predicted_waypoints.append((next_action['x'], next_action['z']))
            if (next_action['x'], next_action['z']) == current_nav_point:
                progress_indicator += 1
                complete_goals += 1
                fail = 0
                actseq_list = [1]
                dist_list = [0]
                print("Direct Success!")
            elif abs(next_action['x'] - current_nav_point[0]) + abs(next_action['z'] - current_nav_point[1]) < 0.5:
                # print(next_action['x'], next_action['z'], current_nav_point)
                progress_indicator += 1
                complete_goals += 1
                fail = 0
                actseq_list = [1]
                dist_list = [0]
                print("Approximate Success!")
            else:
                fail+=1
                # if not event.metadata['lastActionSuccess']:
                #     print("Cannot move to here")
                #     return predicted_waypoints, -1

        return predicted_waypoints, complete_goals, total_goals

    def get_panorama_image(self, env, event):
        yaw = round(event.metadata["agent"]['cameraHorizon'])
        initial_agent= event.metadata["agent"]
        # event = env.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=['z'], rotation=rot['y'], horizon=45.0, tempRenderChange= True, forceAction=True))

        if yaw > 30:
            repeat_lookup = math.ceil((yaw-30)/30)
            for i in range(repeat_lookup):
                event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
        elif yaw < 30:
            repeat_lookdown = math.ceil((30-yaw)/30)
            for i in range(repeat_lookdown):
                event = env.step_panorama({'action': 'LookDown', 'forceAction': True})

        if not event.metadata['lastActionSuccess']:
            yaw = round(event.metadata["agent"]['cameraHorizon'])
            pos = event.metadata["agent"]["position"]
            rot = event.metadata["agent"]["rotation"]
            event = env.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=['z'], rotation=rot['y'], horizon=30.0, forceAction=True))

        rgb_image, mask_image, depth_image, color_to_obj_id_type, last_event = self.panorama_action(env, event)

        teleport_action = {
            'action': 'TeleportFull',
            'rotation': initial_agent["rotation"],
            'x': initial_agent["position"]['x'],
            'z': initial_agent["position"]['z'],
            'y': initial_agent["position"]['y'],
            'horizon': initial_agent["cameraHorizon"],
            'tempRenderChange': True,
            'renderNormalsImage': False,
            'forceAction': True
        }
        env.step(teleport_action)

        return rgb_image, depth_image

    def panorama_action(self, env, event):
        color_to_obj_id_type = {}

        bc = event.frame[:, :, ::-1]
        bc_mask = event.instance_segmentation_frame
        bc_depth = event.depth_frame * 255 / 5000
        bc_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            bc_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
        mc = event.frame[:, :, ::-1]
        mc_mask = event.instance_segmentation_frame
        mc_depth = event.depth_frame * 255 / 5000
        mc_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            mc_meta[str(color)] = object_id

        c_img = np.concatenate((mc, bc), axis=0)
        c_mask = np.concatenate((mc_mask, bc_mask), axis=0)
        c_depth = np.concatenate((mc_depth, bc_depth), axis=0)

        event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})
        ml = event.frame[:, :, ::-1]
        ml_mask = event.instance_segmentation_frame
        ml_depth = event.depth_frame * 255 / 5000
        ml_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            ml_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
        bl = event.frame[:, :, ::-1]
        bl_mask = event.instance_segmentation_frame
        bl_depth = event.depth_frame * 255 / 5000
        bl_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            bl_meta[str(color)] = object_id

        l_img = np.concatenate((ml, bl), axis=0)
        l_mask = np.concatenate((ml_mask, bl_mask), axis=0)
        l_depth = np.concatenate((ml_depth, bl_depth), axis=0)

        event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
        event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
        br = event.frame[:, :, ::-1]
        br_mask = event.instance_segmentation_frame
        br_depth = event.depth_frame * 255 / 5000
        br_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            br_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
        mr = event.frame[:, :, ::-1]
        mr_mask = event.instance_segmentation_frame
        mr_depth = event.depth_frame * 255 / 5000
        mr_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            mr_meta[str(color)] = object_id

        r_img = np.concatenate((mr, br), axis=0)
        r_mask = np.concatenate((mr_mask, br_mask), axis=0)
        r_depth = np.concatenate((mr_depth, br_depth), axis=0)

        event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
        mb = event.frame[:, :, ::-1]
        mb_mask = event.instance_segmentation_frame
        mb_depth = event.depth_frame * 255 / 5000
        mb_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            mb_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
        bb = event.frame[:, :, ::-1]
        bb_mask = event.instance_segmentation_frame
        bb_depth = event.depth_frame * 255 / 5000
        bb_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            bb_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})
        event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})

        b_img = np.concatenate((mb, bb), axis=0)
        b_mask = np.concatenate((mb_mask, bb_mask), axis=0)
        b_depth = np.concatenate((mb_depth, bb_depth), axis=0)

        rgb_image = np.concatenate((l_img, c_img, r_img, b_img), axis=1)
        mask_image = np.concatenate((l_mask, c_mask, r_mask, b_mask), axis=1)
        depth_image = np.concatenate((l_depth, c_depth, r_depth, b_depth), axis=1)

        color_to_obj_id_type = {
            'bc': bc_meta, 'mc': mc_meta, 
            'bl': bl_meta, 'ml': ml_meta,
            'br': br_meta, 'mr': mr_meta, 
            'bb': bb_meta, 'mb': mb_meta,
        }
        rgb_list = [ml, mc, mr, mb, bl, bc, br, bb]
        
        return rgb_list, mask_image, depth_image, color_to_obj_id_type, event