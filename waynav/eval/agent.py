import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from waynav.gen.utils.py_util import walklevel
from waynav.env.thor_env import ThorEnv
from waynav.data.dataset import Panorama_Dataset
from transformers import AutoTokenizer
import math

class Eval_Agent(object):
    def __init__(self, args, model):
        data_path = args.data_path
        save_path = args.save_path
        split = data_path.split('/')[-1]
        waypoint_path = '/mnt/alfworld/data/panorama_' + split
        self.args = args
        self.traj_list = []
        self.waypoint_list = []
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
                if not os.path.isfile(json_file) or json_file in self.finished_list:
                    continue
                
                waypoint_dir = os.path.join(waypoint_path, json_file.split('trial')[0].split('-')[-1], dir_name.split('/')[-1])
                waypoint_file = os.path.join(waypoint_dir, 'traj.json')
                if not os.path.isfile(waypoint_file):
                    continue

                self.waypoint_list.append(waypoint_file)
                self.traj_list.append(json_file)

    def run(self):
        env = ThorEnv(player_screen_width=300, player_screen_height=300)
        traj_list = self.traj_list
        finished_list = self.finished_list
        waypoint_list = self.waypoint_list

        direct_success = 0
        approx_success = 0
        fail = 0 
        while len(traj_list) > 0:
            json_file = traj_list.pop()
            waypoint_file = waypoint_list.pop()

            print ("(%d Left) Evaluating: %s" % (len(traj_list), json_file))
            try:
                predict_waypoint, success = self.predict_waypoint(env, json_file, waypoint_file)
                # self.plot_waypoint(waypoint_file, predict_waypoint)
                if success == 1:
                    direct_success += 1
                elif success == 0.5:
                    approx_success += 1
                else:
                    fail += 1
                finished_list.append(json_file)
                

            except Exception as e:
                import traceback
                traceback.print_exc()
                print ("Error: " + repr(e))

        env.stop()
        print("Finished.")
        print("Direct Success: %d, Approx Success: %d, Fail: %d"%(direct_success, approx_success, fail))

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

    def predict_waypoint(self, env, json_file, waypoint_file):
        device='cuda:0'
        self.model.to(device)

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
        progress_indicator = 1
        predicted_waypoints = []
        while fail < 2:
            last_event = event
            current_nav_point = \
            (waypoint_data['traj']['x'][navpoint_idx[progress_indicator]], waypoint_data['traj']['z'][navpoint_idx[progress_indicator]])
            current_instruction = waypoint_data['instructions'][progress_indicator-1]

            rgb_image, depth_image = self.get_panorama_image(env, event)
            input_ids, rgb_list, depth_list, panorama_angle, panorama_rotation = \
            Panorama_Dataset.process_test_time_data(self.tokenizer, rgb_image, depth_image, current_instruction)
            
            input_ids['input_ids'] = input_ids['input_ids'].to(device)
            rgb_list = rgb_list.to(device)
            depth_list = depth_list.to(device)
            panorama_angle = panorama_angle.to(device)
            panorama_rotation = panorama_rotation.to(device)
            
            next_action_list = self.model.predict_coordinate(input_ids, rgb_list, depth_list, panorama_angle, panorama_rotation, [last_event.metadata['agent']])
            next_action = next_action_list[0]
            predicted_waypoints.append((next_action['x'], next_action['z']))
            if (next_action['x'], next_action['z']) == current_nav_point:
                print("Direct Success!")
                return predicted_waypoints, 1
            elif abs(next_action['x'] - current_nav_point[0]) + abs(next_action['z'] - current_nav_point[1]) < 0.5:
                print(next_action['x'], next_action['z'], current_nav_point)
                print("Approximate Success!")
                return predicted_waypoints, 0.5
            else:
                event = env.step(next_action)
                fail+=1
                if not event.metadata['lastActionSuccess']:
                    print("Cannot teleport to here")
                    return predicted_waypoints, 0

        return predicted_waypoints, 0

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

        event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
        tc = event.frame[:, :, ::-1]
        tc_mask = event.instance_segmentation_frame
        tc_depth = event.depth_frame * 255 / 5000
        tc_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            tc_meta[str(color)] = object_id

        c_img = np.concatenate((tc, mc, bc), axis=0)
        c_mask = np.concatenate((tc_mask, mc_mask, bc_mask), axis=0)
        c_depth = np.concatenate((tc_depth, mc_depth, bc_depth), axis=0)

        event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})
        tl = event.frame[:, :, ::-1]
        tl_mask = event.instance_segmentation_frame
        tl_depth = event.depth_frame * 255 / 5000
        tl_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            tl_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
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

        l_img = np.concatenate((tl, ml, bl), axis=0)
        l_mask = np.concatenate((tl_mask, ml_mask, bl_mask), axis=0)
        l_depth = np.concatenate((tl_depth, ml_depth, bl_depth), axis=0)

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

        event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
        tr = event.frame[:, :, ::-1]
        tr_mask = event.instance_segmentation_frame
        tr_depth = event.depth_frame * 255 / 5000
        tr_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            tr_meta[str(color)] = object_id

        r_img = np.concatenate((tr, mr, br), axis=0)
        r_mask = np.concatenate((tr_mask, mr_mask, br_mask), axis=0)
        r_depth = np.concatenate((tr_depth, mr_depth, br_depth), axis=0)

        event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
        tb = event.frame[:, :, ::-1]
        tb_mask = event.instance_segmentation_frame
        tb_depth = event.depth_frame * 255 / 5000
        tb_meta = {}
        for color, object_id in env.last_event.color_to_object_id.items():
            tb_meta[str(color)] = object_id

        event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
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

        b_img = np.concatenate((tb, mb, bb), axis=0)
        b_mask = np.concatenate((tb_mask, mb_mask, bb_mask), axis=0)
        b_depth = np.concatenate((tb_depth, mb_depth, bb_depth), axis=0)

        rgb_image = np.concatenate((l_img, c_img, r_img, b_img), axis=1)
        mask_image = np.concatenate((l_mask, c_mask, r_mask, b_mask), axis=1)
        depth_image = np.concatenate((l_depth, c_depth, r_depth, b_depth), axis=1)

        color_to_obj_id_type = {
            'bc': bc_meta, 'mc': mc_meta, 'tc': tc_meta,
            'bl': bl_meta, 'ml': ml_meta, 'tl': tl_meta,
            'br': br_meta, 'mr': mr_meta, 'tr': tr_meta,
            'bb': bb_meta, 'mb': mb_meta, 'tb': tb_meta
        }
        return rgb_image, mask_image, depth_image, color_to_obj_id_type, event