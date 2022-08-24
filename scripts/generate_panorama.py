import os
import sys
import json
import glob
import time
import copy
import random
import shutil
import math
import argparse
import threading
from multiprocessing import Pool
from collections import deque
from turtle import clear
import cv2
import numpy as np

import waynav.gen
import waynav.gen.constants as constants
from waynav.gen.utils.py_util import walklevel
from waynav.env.thor_env import ThorEnv
from PIL import Image

TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

IMAGES_FOLDER = "images"
MASKS_FOLDER = "masks"
META_FOLDER = "meta"
DEPTH_FOLDER = "depth"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

scene_dict = {'kitchen':0, 'bedroom':0, 'bathroom':0, 'living':0}


# def get_image_index(save_path):
#     return len(glob.glob(save_path + '/*.png'))


# def save_image_with_delays(env, action,
#                            save_path, direction=constants.BEFORE):
#     im_ind = get_image_index(save_path)
#     counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
#     for i in range(counts):
#         save_image(env.last_event, save_path)
#         env.noop()
#     return im_ind


# def save_image(event, save_path):
#     # rgb
#     rgb_save_path = os.path.join(save_path, IMAGES_FOLDER)
#     rgb_image = event.frame[:, :, ::-1]

#     # masks
#     mask_save_path = os.path.join(save_path, MASKS_FOLDER)
#     mask_image = event.instance_segmentation_frame

#     # depth
#     depth_save_path = os.path.join(save_path, DEPTH_FOLDER)
#     depth_image = event.depth_frame * 255 / 5000

#     im_ind = len(os.listdir(rgb_save_path))
#     status1 = cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image)
#     status2 = cv2.imwrite(mask_save_path + '/%09d.png' % im_ind, mask_image)
#     cv2.imwrite(depth_save_path + '/%09d.png' % im_ind, depth_image)    
    
#     return im_ind


# def save_images_in_events(events, root_dir):
#     for event in events:
#         save_image(event, root_dir)


# def clear_and_create_dir(path):
#     # if os.path.exists(path):
#     #     shutil.rmtree(path)
#     if not os.path.exists(path):
#         os.makedirs(path)


def get_scene_type(scene_num):
    if scene_num < 100:
        scene_dict['kitchen']+=1
        return 'kitchen'
    elif scene_num < 300:
        scene_dict['living']+=1
        return 'living'
    elif scene_num < 400:
        scene_dict['bedroom']+=1
        return 'bedroom'
    else:
        scene_dict['bathroom']+=1
        return 'bathroom'


def get_openable_points(traj_data):
    scene_num = traj_data['scene']['scene_num']
    openable_json_file = os.path.join(alfworld.gen.__path__[0], 'layouts/FloorPlan%d-openable.json' % scene_num)
    with open(openable_json_file, 'r') as f:
        openable_points = json.load(f)
    return openable_points


# def explore_scene(env, traj_data, root_dir):
#     '''
#     Use pre-computed openable points from ALFRED to store receptacle locations
#     '''
#     openable_points = get_openable_points(traj_data)
#     agent_height = env.last_event.metadata['agent']['position']['y']
#     for recep_id, point in openable_points.items():
#         recep_class = recep_id.split("|")[0]
#         action = {'action': 'TeleportFull',
#                   'x': point[0],
#                   'y': agent_height,
#                   'z': point[1],
#                   'rotateOnTeleport': False,
#                   'rotation': point[2],
#                   'horizon': point[3]}
#         event = env.step(action)
#         save_frame(env, event, root_dir)

def augment_traj(env, json_file):
    # load json data
    splitted = json_file.split('/')[-2]
    print(splitted)
    with open(json_file) as f:
        traj_data = json.load(f)


    # fresh images list
    traj_data['images'] = list()

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    with lock:
        scene_type = get_scene_type(scene_num)
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    root_dir = os.path.join(args.save_path, str(scene_num), splitted)

    imgs_dir = os.path.join(root_dir, IMAGES_FOLDER)
    mask_dir = os.path.join(root_dir, MASKS_FOLDER)
    meta_dir = os.path.join(root_dir, META_FOLDER)
    depth_dir = os.path.join(root_dir, DEPTH_FOLDER)

    clear_and_create_dir(root_dir)
    clear_and_create_dir(imgs_dir)
    clear_and_create_dir(mask_dir)
    clear_and_create_dir(meta_dir)
    clear_and_create_dir(depth_dir)

    # explore_scene(env, traj_data, root_dir)
    x_list = []
    z_list = []
    rotation_list = []
    angle_list = []
    event = env.step(dict(traj_data['scene']['init_action']))
    # save_frame(env, event, root_dir)
    # coord = event.metadata["agent"]['position']
    # camera_angle = event.metadata["agent"]["cameraHorizon"]
    # print(camera_angle)
    # action_count = 0
    # x_list.append(coord['x'])
    # z_list.append(coord['z'])
    # key_x.append(coord['x'])
    # key_z.append(coord['z'])
    # finish_x.append(coord['x'])
    # finish_z.append(coord['z'])
    # waypoint_ind.append(action_count)
    # print("Task: %s" % (traj_data['template']['task_desc']))

    # setup task
    env.set_task(traj_data, args, reward_type='dense')
    rewards = []
    prev_high_pddl_ind = 0
    navigation_point = []
    instructions = deque(traj_data['turk_annotations']['anns'][0]['high_descs'])
    angle = 45

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        # next cmd under the current hl_action
        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]
        discrete_action = hl_action['discrete_action']['action']
        # remove unnecessary keys
        cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if k in cmd}

        if ll_idx == 1:
            current_goal = instructions.popleft()
            x_list.append(coord['x'])
            z_list.append(coord['z'])
            rotation_list.append(rotation)
            angle_list.append(angle)
            if discrete_action == 'GotoLocation':
                save_panorama(env, event, root_dir)
                navigation_point.append(ll_action['high_idx'])
            else:
                save_frame(env, event, root_dir)
        
        if prev_high_pddl_ind != (ll_action['high_idx']):
            # high_pddl_change.append(action_count)
            prev_high_pddl_ind = ll_action['high_idx']
            x_list.append(coord['x'])
            z_list.append(coord['z'])
            angle_list.append(angle)
            rotation_list.append(rotation)
            current_goal = instructions.popleft()
            if discrete_action == 'GotoLocation':
                save_panorama(env, event, root_dir)
                navigation_point.append(ll_action['high_idx'])
            else:
                save_frame(env, event, root_dir)

        event = env.step(cmd)
        coord = event.metadata["agent"]['position']
        rotation = event.metadata["agent"]["rotation"]['y']
        angle = event.metadata["agent"]['cameraHorizon']
            # x_list.append(coord['x'])
            # z_list.append(coord['z'])
            # key_x.append(coord['x'])
            # key_z.append(coord['z'])
            # waypoint_ind.append(action_count)
        if not event.metadata['lastActionSuccess']:
            raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))

    x_list.append(coord['x'])
    z_list.append(coord['z'])
    angle_list.append(angle)
    rotation_list.append(rotation)
        
    with open(os.path.join(root_dir, 'traj.json'), 'w') as f:
        data_structure = {'traj':{'x': x_list, 'z': z_list, 'angle': angle_list, 'rotation': rotation_list}, 'instructions':traj_data['turk_annotations']['anns'][0]['high_descs'],\
            'navigation_point': navigation_point }
        json.dump(data_structure, f)

# def save_panorama(env, event, root_dir):
#     rgb_save_path = os.path.join(root_dir, IMAGES_FOLDER)
#     mask_save_path = os.path.join(root_dir, MASKS_FOLDER)
#     depth_save_path = os.path.join(root_dir, DEPTH_FOLDER)
#     im_ind = len(os.listdir(rgb_save_path))
#     yaw = round(event.metadata["agent"]['cameraHorizon'])
#     initial_agent= event.metadata["agent"]
#     # event = env.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=['z'], rotation=rot['y'], horizon=45.0, tempRenderChange= True, forceAction=True))

#     if yaw > 30:
#         repeat_lookup = math.ceil((yaw-30)/30)
#         for i in range(repeat_lookup):
#             event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
#     elif yaw < 30:
#         repeat_lookdown = math.ceil((30-yaw)/30)
#         for i in range(repeat_lookdown):
#             event = env.step_panorama({'action': 'LookDown', 'forceAction': True})

#     if not event.metadata['lastActionSuccess']:
#         yaw = round(event.metadata["agent"]['cameraHorizon'])
#         pos = event.metadata["agent"]["position"]
#         rot = event.metadata["agent"]["rotation"]
#         event = env.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=['z'], rotation=rot['y'], horizon=30.0, forceAction=True))

#     rgb_image, mask_image, depth_image, color_to_obj_id_type, last_event = get_panorama(env, event)
#     cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image)
#     cv2.imwrite(mask_save_path + '/%09d.png' % im_ind, mask_image)
#     cv2.imwrite(depth_save_path + '/%09d.png' % im_ind, depth_image)
    
#     meta_file = os.path.join(root_dir, META_FOLDER, "%09d.json" % im_ind)
#     with open(meta_file, 'w') as f:
#         json.dump(color_to_obj_id_type, f)

#     # event = env.step(dict(action='TeleportFull', x=pos['x'], y=pos['y'], z=['z'], rotation=rot['y'], horizon=yaw, tempRenderChange= True, forceAction=True))

#     # yaw = round(last_event.metadata["agent"]['cameraHorizon'])

#     # if yaw > original_yaw:
#     #     repeat_lookup = math.ceil((yaw-original_yaw)/30)
#     #     for i in range(repeat_lookup):
#     #         event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
#     #         yaw -= 30
#     # elif yaw < original_yaw:
#     #     repeat_lookdown = math.ceil((original_yaw-yaw)/30)
#     #     for i in range(repeat_lookdown):
#     #         event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
#     #         yaw += 30

#     # if yaw > original_yaw:
#     #     event = env.step({'action': 'LookUp', 'forceAction': True})
#     #     yaw = yaw - 15
#     # elif yaw < original_yaw:
#     #     event = env.step({'action': 'LookDown', 'forceAction': True})
#     #     yaw = yaw + 15

#     # assert yaw == original_yaw
#     teleport_action = {
#         'action': 'TeleportFull',
#         'rotation': initial_agent["rotation"],
#         'x': initial_agent["position"]['x'],
#         'z': initial_agent["position"]['z'],
#         'y': initial_agent["position"]['y'],
#         'horizon': initial_agent["cameraHorizon"],
#         'tempRenderChange': True,
#         'renderNormalsImage': False,
#         'forceAction': True
#     }
#     env.step(teleport_action)

# def save_frame(env, event, root_dir):
#     im_idx = save_image(event, root_dir)
#     # store color to object type dictionary
#     color_to_obj_id_type = {}
#     all_objects = env.last_event.metadata['objects']
#     for color, object_id in env.last_event.color_to_object_id.items():
#         color_to_obj_id_type[str(color)] = object_id
#     meta_file = os.path.join(root_dir, META_FOLDER, "%09d.json" % im_idx)
#     with open(meta_file, 'w') as f:
#         json.dump(color_to_obj_id_type, f)


# def get_panorama(env, event):
#     color_to_obj_id_type = {}

#     bc = event.frame[:, :, ::-1]
#     bc_mask = event.instance_segmentation_frame
#     bc_depth = event.depth_frame * 255 / 5000
#     bc_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         bc_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
#     mc = event.frame[:, :, ::-1]
#     mc_mask = event.instance_segmentation_frame
#     mc_depth = event.depth_frame * 255 / 5000
#     mc_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         mc_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
#     tc = event.frame[:, :, ::-1]
#     tc_mask = event.instance_segmentation_frame
#     tc_depth = event.depth_frame * 255 / 5000
#     tc_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         tc_meta[str(color)] = object_id

#     c_img = np.concatenate((tc, mc, bc), axis=0)
#     c_mask = np.concatenate((tc_mask, mc_mask, bc_mask), axis=0)
#     c_depth = np.concatenate((tc_depth, mc_depth, bc_depth), axis=0)

#     event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})
#     tl = event.frame[:, :, ::-1]
#     tl_mask = event.instance_segmentation_frame
#     tl_depth = event.depth_frame * 255 / 5000
#     tl_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         tl_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
#     ml = event.frame[:, :, ::-1]
#     ml_mask = event.instance_segmentation_frame
#     ml_depth = event.depth_frame * 255 / 5000
#     ml_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         ml_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
#     bl = event.frame[:, :, ::-1]
#     bl_mask = event.instance_segmentation_frame
#     bl_depth = event.depth_frame * 255 / 5000
#     bl_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         bl_meta[str(color)] = object_id

#     l_img = np.concatenate((tl, ml, bl), axis=0)
#     l_mask = np.concatenate((tl_mask, ml_mask, bl_mask), axis=0)
#     l_depth = np.concatenate((tl_depth, ml_depth, bl_depth), axis=0)

#     event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
#     event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
#     br = event.frame[:, :, ::-1]
#     br_mask = event.instance_segmentation_frame
#     br_depth = event.depth_frame * 255 / 5000
#     br_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         br_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
#     mr = event.frame[:, :, ::-1]
#     mr_mask = event.instance_segmentation_frame
#     mr_depth = event.depth_frame * 255 / 5000
#     mr_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         mr_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookUp', 'forceAction': True})
#     tr = event.frame[:, :, ::-1]
#     tr_mask = event.instance_segmentation_frame
#     tr_depth = event.depth_frame * 255 / 5000
#     tr_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         tr_meta[str(color)] = object_id

#     r_img = np.concatenate((tr, mr, br), axis=0)
#     r_mask = np.concatenate((tr_mask, mr_mask, br_mask), axis=0)
#     r_depth = np.concatenate((tr_depth, mr_depth, br_depth), axis=0)

#     event = env.step_panorama({'action': 'RotateRight', 'forceAction': True})
#     tb = event.frame[:, :, ::-1]
#     tb_mask = event.instance_segmentation_frame
#     tb_depth = event.depth_frame * 255 / 5000
#     tb_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         tb_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
#     mb = event.frame[:, :, ::-1]
#     mb_mask = event.instance_segmentation_frame
#     mb_depth = event.depth_frame * 255 / 5000
#     mb_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         mb_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'LookDown', 'forceAction': True})
#     bb = event.frame[:, :, ::-1]
#     bb_mask = event.instance_segmentation_frame
#     bb_depth = event.depth_frame * 255 / 5000
#     bb_meta = {}
#     for color, object_id in env.last_event.color_to_object_id.items():
#         bb_meta[str(color)] = object_id

#     event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})
#     event = env.step_panorama({'action': 'RotateLeft', 'forceAction': True})

#     b_img = np.concatenate((tb, mb, bb), axis=0)
#     b_mask = np.concatenate((tb_mask, mb_mask, bb_mask), axis=0)
#     b_depth = np.concatenate((tb_depth, mb_depth, bb_depth), axis=0)

#     rgb_image = np.concatenate((l_img, c_img, r_img, b_img), axis=1)
#     mask_image = np.concatenate((l_mask, c_mask, r_mask, b_mask), axis=1)
#     depth_image = np.concatenate((l_depth, c_depth, r_depth, b_depth), axis=1)

#     color_to_obj_id_type = {
#         'bc': bc_meta, 'mc': mc_meta, 'tc': tc_meta,
#         'bl': bl_meta, 'ml': ml_meta, 'tl': tl_meta,
#         'br': br_meta, 'mr': mr_meta, 'tr': tr_meta,
#         'bb': bb_meta, 'mb': mb_meta, 'tb': tb_meta
#     }
#     return rgb_image, mask_image, depth_image, color_to_obj_id_type, event





def run():
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,
                  player_screen_height=IMAGE_HEIGHT)

    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print ("(%d Left) Augmenting: %s" % (len(traj_list), json_file))
        try:
            augment_traj(env, json_file)
            lock.acquire()
            finished_list.append(json_file)
            lock.release()
            

        except Exception as e:
            import traceback
            traceback.print_exc()
            print ("Error: " + repr(e))

    env.stop()
    print("Finished.")

def mp_run(_):
    return run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/mnt/alfworld/data/json_2.1.1/valid_unseen")
    parser.add_argument('--save_path', type=str, default="/mnt/alfworld/data/test_waynav")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true')
    parser.add_argument('--time_delays', dest='time_delays', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--reward_config', type=str, default='alfworld/agents/config/rewards.json')
    args = parser.parse_args()


# cache
cache_file = os.path.join(args.save_path, "cache.json")
if os.path.isfile(cache_file):
    with open(cache_file, 'r') as f:
        finished_jsons = json.load(f)
else:
    finished_jsons = {'finished': []}

# make a list of all the traj_data json files
data_path = os.path.expandvars(args.data_path)
for dir_name, subdir_list, file_list in walklevel(data_path, level=2):
    if "trial_" in dir_name:
        json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
        if not os.path.isfile(json_file) or json_file in finished_jsons['finished']:
            continue
        traj_list.append(json_file)

# random shuffle
if args.shuffle:
    random.shuffle(traj_list)

total_episodes = len(traj_list)
finished_list = []
start = time.time()
counter = 0
# run()
# start threads
threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run)
    threads.append(thread)
    thread.start()
    time.sleep(1)

for i in range(args.num_threads):
    threads[i].join()

cache_file = os.path.join(args.save_path, "cache.json")
with open(cache_file, 'w') as f:
    json.dump({'finished': finished_list}, f)

duration = time.time()-start
print(f"Cost {duration} seconds")
print(scene_dict)
print("Total: %d; Finished: %d"%(total_episodes, len(finished_list)))
