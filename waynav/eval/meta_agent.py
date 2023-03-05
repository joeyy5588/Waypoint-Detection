import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
from waynav.gen.utils.py_util import walklevel
from waynav.gen.constants import OBJECTS_DETECTOR, STATIC_RECEPTACLES
from waynav.env.thor_env import ThorEnv
from transformers import AutoTokenizer, AutoConfig
from .detection_utils import Detection_Helper
from waynav.model import VLN_Navigator, ROI_Waypoint_Predictor
from .navigation_utils import Navigation_Helper
import math
import torch

class Eval_Subpolicy_Agent(object):
    def __init__(self, args):
        # Path for testing data
        data_path = args.data_path
        save_path = args.save_path
        split = data_path.split('/')[-1]
        split_fn = json.load(open('/data/joey/alfred_metadata/oct21.json'))
        inst_dict = json.load(open('/data/joey/alfred_metadata/'+split+'/inst_dict.json'))
        if 'valid' in split:
            self.subpolicy＿dict = json.load(open('/data/joey/alfred_metadata/'+split+'/subpolicy_sidestep.json'))
        else:
            self.subpolicy_dict = None

        self.inst2type = inst_dict['type']
        self.inst2action = inst_dict['action']
        self.args = args
        self.use_gt_nav = False
        # self.use_gt_lang = True
        self.use_gt_mask = False
        self.use_gt_subpolicy = True
        self.traj_list = []
        self.subpolicy_list = []
        # self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.object_model = Detection_Helper(args, args.object_model_path)
        self.recep_model = Detection_Helper(args, args.recep_model_path, object_types='receptacles')

        self.subpolicy_model = Navigation_Helper(args, 'cuda:2', 'high')
        self.ll_model = Navigation_Helper(args, 'cuda:2', 'low')
        self.open_mask = None

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
                success, gc, gn = self.execute_task(env, json_file)
                trial_num += 1
                trial_success += success
                goal_num += gn
                goal_success += gc

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

    def execute_task(self, env, traj_data, r_idx=0):
        device='cuda:0'
        self.object_model.to_device(device)
        self.recep_model.to_device(device)

        traj_data = json.load(open(traj_data))
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

        instructions = traj_data['turk_annotations']['anns'][r_idx]['high_descs']

        for inst_idx, inst in enumerate(instructions):
            inst_type = self.inst2type[inst.lower().strip()]
            if inst_type == 'navigation':
                self.do_navigation(env, inst, traj_data, inst_idx)
            elif inst_type == 'interaction':
                self.do_interaction(env, inst, traj_data, inst_idx)

        if env.get_goal_satisfied():
            print("Goal Reached")
            success = 1
        else:
            print("Goal Not Reached")
            success = 0

        goal_success, goal_num = env.get_goal_conditions_met()

        print(success, goal_success, goal_num)
        return success, goal_success, goal_num

    def do_navigation(self, env, inst, traj_data, inst_idx):
        if self.use_gt_nav:
            for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
                if ll_action['high_idx'] == inst_idx:
                    cmd = ll_action['api_action']
                    cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if k in cmd}
                    event = env.step(cmd)
                    if not event.metadata['lastActionSuccess']:
                        raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))
        else:
            subpolicy_idx = 0
            traj_key = str(traj_data['scene']['scene_num']) + '/' + traj_data['task_id']

            if self.use_gt_subpolicy:
                subpolicy = self.subpolicy_dict[traj_key].pop(0)
                subpolicy = subpolicy.split(' ')
                for s in range(len(subpolicy)//2):
                    self.subpolicy_list.append(subpolicy[2*s] + ' ' + subpolicy[2*s+1])
            else:
                print('GT:', self.subpolicy_dict[traj_key].pop(0))
                rgb_image = self.get_panorama_image(env, env.last_event)
                with torch.no_grad():
                    patch_feat = self.object_model.extract_cnn_features(rgb_image)
                    obj_cls, _, _  = self.object_model.extract_roi_features(rgb_image)
                    recep_cls, _, _ = self.recep_model.extract_roi_features(rgb_image)
                pred_subpolicy = self.subpolicy_model.predict(inst, traj_data, patch_feat, obj_cls, recep_cls)
                self.subpolicy_list += (pred_subpolicy)

            subpolicy_count = 0
            while self.subpolicy_list:
                curr_subpolicy = self.subpolicy_list.pop(0)
                if self.subpolicy_list:
                    next_subpolicy = self.subpolicy_list[0]
                else:
                    next_subpolicy = "interaction"
                print('Curr subpolicy:', curr_subpolicy, self.subpolicy_list)
                subpolicy_count += 1
                next_subpolicy = False
                first_ll_action = True
                ll_action_count = 0
                while not next_subpolicy and ll_action_count < 30:
                    if curr_subpolicy == 'turn left':
                        success, event, target_instance_id, err, _ = env.va_interact('RotateLeft')
                        next_subpolicy = True
                    elif curr_subpolicy == 'turn right':
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        next_subpolicy = True
                    elif curr_subpolicy == 'turn around':
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        next_subpolicy = True
                    elif curr_subpolicy == 'step back':
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                        success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        next_subpolicy = True
                    elif curr_subpolicy == 'step left':
                        success, event, target_instance_id, err, _ = env.va_interact('RotateLeft')
                        success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                        success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        next_subpolicy = True
                    elif curr_subpolicy == 'step right':
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                        success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                        success, event, target_instance_id, err, _ = env.va_interact('RotateLeft')
                        next_subpolicy = True
                    elif curr_subpolicy == 'face left' and first_ll_action:
                        success, event, target_instance_id, err, _ = env.va_interact('RotateLeft')
                        first_ll_action = False
                    elif curr_subpolicy == 'face right' and first_ll_action:
                        success, event, target_instance_id, err, _ = env.va_interact('RotateRight')
                        first_ll_action = False

                    else:
                        rgb_image = self.get_panorama_image(env, env.last_event)
                        with torch.no_grad():
                            patch_feat = self.object_model.extract_cnn_features(rgb_image)
                            obj_cls, _, _  = self.object_model.extract_roi_features(rgb_image)
                            recep_cls, _, _ = self.recep_model.extract_roi_features(rgb_image)
                        pred_action = self.ll_model.predict(inst, traj_data, patch_feat, obj_cls, recep_cls, curr_subpolicy, next_subpolicy)
                        if pred_action == 'Interaction':
                            next_subpolicy = True
                        else:
                            success, event, target_instance_id, err, _ = env.va_interact('MoveAhead')
                            ll_action_count += 1
        return

    def do_interaction(self, env, inst, traj_data, inst_idx):
        debug_dir = '/data/joey/alfred_metadata/debug'
        action_seq = self.inst2action[inst.lower().strip()]
        action_len = len(action_seq)
        last_action = ''
        for i in range(len(action_seq) // 2):
            curr_action = action_seq[i * 2]
            target_object = action_seq[i * 2 + 1]
            if last_action == 'OpenObject' and curr_action == 'PutObject':
                interaction_mask = self.open_mask
            elif curr_action == 'CloseObject':
                interaction_mask = self.open_mask
            else:
                if self.use_gt_mask:
                    interaction_mask = self.get_interaction_mask_gt(env, target_object)
                else:
                    interaction_mask = self.get_interaction_mask(env, target_object)
            success, event, target_instance_id, err, _ = env.va_interact(curr_action, interact_mask=interaction_mask)
            if success and curr_action == 'OpenObject':
                self.open_mask = copy.deepcopy(interaction_mask)
            if not success:
                print(target_object, curr_action, err)
            # if not success and (target_object=='Cabinet' or target_object=='Drawer'):
            #     img_idx = len(os.listdir(debug_dir))
            #     cv2.imwrite(os.path.join(debug_dir, '%d_view.png' % img_idx), env.last_event.frame)
            #     cv2.imwrite(os.path.join(debug_dir, '%d_mask.png' % img_idx), interaction_mask*255)
            last_action = curr_action


        return

    def get_interaction_mask_gt(self, env, object_type):
        mask = np.zeros((300, 300))
        found = False
        for k, v in env.last_event.instance_masks.items():
            category = k.split('|')[0]
            category_last = k.split('|')[-1]
            if 'Sliced' in category_last:
                category = category + 'Sliced'
            if 'Sink' in category and 'SinkBasin' in category_last:
                category =  'SinkBasin' 
            if 'Bathtub' in category and 'BathtubBasin' in category_last:
                category =  'BathtubBasin'
            if category == object_type:
                found = True
                mask = v
                break
        if found:
            # print('found', category, object_type)
            return mask
        else:
            # print('not found')
            if object_type in ['ButterKnife', 'Knife']:
                object_type = ['ButterKnife', 'Knife']
            elif object_type in ['Mug', 'Cup']:
                object_type = ['Mug', 'Cup']
            elif object_type in ['FloorLamp', 'DeskLamp']:
                object_type = ['FloorLamp', 'DeskLamp']
            elif object_type in ['Spoon', 'Ladle']:
                object_type = ['Spoon', 'Ladle']
            elif object_type in ['DiningTable', 'SideTable', 'Dresser', 'CounterTop']:
                object_type = ['DiningTable', 'SideTable', 'Dresser', 'CounterTop']

            for k, v in env.last_event.instance_masks.items():
                category = k.split('|')[0]
                category_last = k.split('|')[-1]
                if 'Sliced' in category_last:
                    category = category + 'Sliced'
                if 'Sink' in category and 'SinkBasin' in category_last:
                    category =  'SinkBasin' 
                if 'Bathtub' in category and 'BathtubBasin' in category_last:
                    category =  'BathtubBasin'
                if category in object_type:
                    found = True
                    print('found in second search', category, object_type)
                    mask = v

        if np.sum(mask) == 0:
            mask = None
        return mask

    def get_interaction_mask(self, env, object_type):
        mask = np.zeros((300, 300))
        if object_type in OBJECTS_DETECTOR:
            prediction = self.object_model.predict_mask(env.last_event.frame)
        elif object_type in STATIC_RECEPTACLES:
            prediction = self.recep_model.predict_mask(env.last_event.frame)
        else:
            print('object doesnot exist in the detector', object_type)

        scores = prediction['scores']
        pred_masks = prediction['pred_masks']
        pred_classes = prediction['pred_classes']
        for i in range(len(scores)):
            if pred_classes[i] == object_type:
                mask = pred_masks[i].astype(np.float32)
                # print('find', pred_classes[i])
                return mask

        if object_type in ['ButterKnife', 'Knife']:
            object_type = ['ButterKnife', 'Knife']
        elif object_type in ['Mug', 'Cup']:
            object_type = ['Mug', 'Cup']
        elif object_type in ['FloorLamp', 'DeskLamp']:
            object_type = ['FloorLamp', 'DeskLamp']
        elif object_type in ['Spoon', 'Ladle']:
            object_type = ['Spoon', 'Ladle']
        elif object_type in ['DiningTable', 'SideTable', 'Dresser', 'CounterTop']:
            object_type = ['DiningTable', 'SideTable', 'Dresser', 'CounterTop']
        
        for i in range(len(scores)):
            if pred_classes[i] in object_type:
                mask = pred_masks[i].astype(np.float32)
                # print('find', pred_classes[i])
                return mask

        if np.sum(mask) == 0:
            mask = None
        return mask

    def get_max_interaction_mask(self, env, object_type):
        mask = np.zeros((300, 300))
        if object_type in OBJECTS_DETECTOR:
            prediction = self.object_model.predict_mask(env.last_event.frame)
        elif object_type in STATIC_RECEPTACLES:
            prediction = self.recep_model.predict_mask(env.last_event.frame)
        else:
            print('object doesnot exist in the detector', object_type)

        max_area = 0
        scores = prediction['scores']
        pred_masks = prediction['pred_masks']
        pred_classes = prediction['pred_classes']
        for i in range(len(scores)):
            if pred_classes[i] == object_type:
                # print('find', pred_classes[i])
                temp_mask = pred_masks[i].astype(np.float32)
                mask_area = np.sum(temp_mask)
                if mask_area > max_area:
                    max_area = mask_area
                    mask = temp_mask
                    print(max_area)

        if max_area != 0:
            return mask

        if object_type in ['ButterKnife', 'Knife']:
            object_type = ['ButterKnife', 'Knife']
        elif object_type in ['Mug', 'Cup']:
            object_type = ['Mug', 'Cup']
        elif object_type in ['FloorLamp', 'DeskLamp']:
            object_type = ['FloorLamp', 'DeskLamp']
        elif object_type in ['Spoon', 'Ladle']:
            object_type = ['Spoon', 'Ladle']
        elif object_type in ['DiningTable', 'SideTable', 'Dresser', 'CounterTop']:
            object_type = ['DiningTable', 'SideTable', 'Dresser', 'CounterTop']
        
        for i in range(len(scores)):
            if pred_classes[i] in object_type:
                temp_mask = pred_masks[i].astype(np.float32)
                mask_area = np.sum(temp_mask)
                if mask_area > max_area:
                    max_area = mask_area
                    mask = temp_mask

        if np.sum(mask) == 0:
            mask = None
        return mask


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

        return rgb_image

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