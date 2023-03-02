import json
from collections import deque
import numpy as np
import os
import re
from torchmetrics import ConfusionMatrix

def compute_confusion_matrix(eval_preds):

    action_labels = eval_preds['action_labels']
    boundary_labels = eval_preds['boundary_labels']
    action_preds = eval_preds['action_preds']
    boundary_preds = eval_preds['boundary_preds']

    action_confmat = ConfusionMatrix(task="multiclass", num_classes=4)
    boundary_confmat = ConfusionMatrix(task="binary", num_classes=2)
    print(action_confmat(action_preds, action_labels))
    print(boundary_confmat(boundary_preds, boundary_labels))
    
def compute_meta_action_metrics(eval_preds):
    preds, labels = eval_preds
    ignore_index = (labels == -100)
    # Teacher-forcing inference or greedy decoding
    # print(preds)
    if isinstance(preds, tuple):
        preds = np.argmax(preds[0], axis=2)
        pad_idx = -100
    else:
        preds = preds[:,1:]
        labels = labels[:,:-1]
        pad_idx = 1
    ignore_index = (labels == pad_idx)
    # print(preds)
    # for i in range(preds.shape[0]):
    #     print(preds[i], labels[i])
    data_num = preds.shape[0]
    equal_logits = ((preds == labels)|ignore_index)
    all_equal = np.sum(np.all(equal_logits, axis=1))
    eval_dict = {"Equal_All": all_equal/data_num}
    for i in range(4):
        ith_equal = np.sum(np.all(equal_logits[:,:i+1], axis=1))
        eval_dict["Equal_"+str(i+1)] = ith_equal/data_num

    re_match, bf_match = compute_meta_action_scores(preds, labels)
    eval_dict["RE_match"] = re_match/data_num
    eval_dict["BF_match"] = bf_match/data_num
    eval_dict["RE_acc_SR"] = eval_dict["RE_match"] + eval_dict["Equal_All"]
    eval_dict["BF_acc_SR"] = eval_dict["RE_match"] + eval_dict["BF_match"] + eval_dict["Equal_All"]
    print("finish evaluation")
    # np.savez("/local1/cfyang/output/subpolicy/inference/unseen_predict_1500.npz", preds=preds, labels=labels)
    # print(eval_dict)
    return eval_dict

def compute_meta_action_scores(preds, labels):
    if preds.shape[0] == 788:
        split = 'seen'
    else:
        split = 'unseen'
    fn = json.load(open('/local1/cfyang/output/subpolicy/inference/'+split+'_fn.json'))
    # subpolicy_to_int = {
    #     'move forward': 3,
    #     'turn left': 4,
    #     'turn right': 5,
    #     'turn around': 6,
    #     'front left': 7,
    #     'front right': 8,
    #     'step back': 9,
    #     'face left': 10,
    #     'face right': 11,
    # }
    subpolicy_to_int = {
        'move forward': 3,
        'turn left': 4,
        'turn right': 5,
        'turn around': 6,
        'side step': 7,
        'step back': 8,
        'face left': 9,
        'face right': 10,
    }
    subpolicy_to_re = {
        '': '',
        'move forward': "m{1,}",
        'turn left': "l{1}",
        'turn right': "r{1}",
        'turn around': "(lm?l)|(rm?r)",
        'side step': "(m*lm{,2}rm*)|(m*rm{,2}lm*)",
        'front left': "(m*lm{,2}rm*)",
        'front right': "(m*rm{,2}lm*)",
        'step back': "(ll|rr)m+(ll|rr)",
        'face left': "(lm{0,2}$)|(l$)",
        'face right': "(rm{0,2}$)|(r$)",
    }
    action_to_alpha = {
        'MoveAhead': 'm',
        'RotateRight': 'r',
        'RotateLeft': 'l',
        'LookUp': '',
        'LookDown': '',
    }

    # change dict key and value
    int_to_subpolicy = dict([(value, key) for key, value in subpolicy_to_int.items()])
    int_to_subpolicy[0] = ''
    int_to_subpolicy[1] = ''
    int_to_subpolicy[2] = ''
    fn_to_str = {}

    exact_match = 0
    approx_match = 0
    re_match = 0
    data_num = 0
    for idx, name in enumerate(fn):
        if idx == preds.shape[0]:
            break
        dir = name.split('images')[0].replace('/data/joey', '/local1/cfyang')
        split_dir = (name.split('/'))
        # load string traj
        anno_dir_traj = os.path.join('/local1/cfyang/alfworld/data/json_2.1.1', 'valid_'+split, split_dir[4], split_dir[5], 'traj_data.json')
        if anno_dir_traj in fn_to_str:
            gt_re = fn_to_str[anno_dir_traj].pop(0)
        else:
            traj = json.load(open(anno_dir_traj))
            ll_actions = traj['plan']['low_actions']
            high_idx_list = [x['high_idx'] for x in ll_actions]
            ll_actseq = [x['api_action']['action'] for x in ll_actions]
            instructions = traj['turk_annotations']['anns'][0]['high_descs']
            high_pddl = traj['plan']['high_pddl']
            high_pddl = [x['discrete_action'] for x in high_pddl]

            prev_idx = 0
            curr_idx = 0
            partitioned_act = []
            for i in range(1,max(high_idx_list)+1):
                curr_idx = high_idx_list.index(i)
                partitioned_act.append(ll_actseq[prev_idx:curr_idx])
                prev_idx = curr_idx
            partitioned_act.append(ll_actseq[prev_idx:])

            nav_set = set(action_to_alpha.keys())
            nav_inst = []
            nav_pddl = []
            string_traj = []
            for i in range(len(partitioned_act)):
                subtask = (partitioned_act[i])
                if not nav_set.isdisjoint(set(subtask)):
                    nav_inst.append(instructions[i])
                    nav_pddl.append(high_pddl[i])
                    subtask = [action_to_alpha[x] for x in subtask]
                    subtask = ''.join(subtask)
                    string_traj.append(subtask)
            gt_re = string_traj.pop(0)
            fn_to_str[anno_dir_traj] = string_traj

        nav_idx = int(name.split('images/')[1].split('.png')[0])
        traj = json.load(open(dir+'traj.json'))
        nav_point = traj['navigation_point']
        instructions = traj['instructions']
        rotation = traj['traj']['rotation']
        rot = round(rotation[nav_idx])
        x_traj = traj['traj']['x']
        z_traj = traj['traj']['z']
        next_nav_idx = nav_idx+1
        while next_nav_idx < max(nav_point)+1:
            if next_nav_idx not in nav_point:
                break
            next_nav_idx+=1

        # print(nav_idx, next_nav_idx, nav_point, len(instructions), len(x_traj))
        # # print(len(x_traj), max(nav_point), traj['instructions'])
        # for i in range(len(instructions)):
        #     print(i, instructions[i])

        delta_x = x_traj[next_nav_idx] - x_traj[nav_idx]
        delta_z = z_traj[next_nav_idx] - z_traj[nav_idx]
        if rot == 0:
            target_x = delta_x
            target_z = delta_z
        elif rot == 90:
            target_x = -delta_z
            target_z = delta_x
        elif rot == 180:
            target_x = -delta_x
            target_z = -delta_z
        elif rot == 270:
            target_x = delta_z
            target_z = -delta_x
        pred_str = [int_to_subpolicy[x] for x in preds[idx]]
        labels_str = [int_to_subpolicy[x] for x in labels[idx]]
        re_str = [subpolicy_to_re[x] for x in pred_str]
        re_str = ''.join(re_str)
        pred_str = ' '.join(pred_str).strip()
        labels_str = ' '.join(labels_str).strip()
        # print(target_x, target_z)
        # print('pred:', pred_str, "GT:", labels_str)
        # print(pred_str, labels_str)
        # print(re_str, gt_re)
        # print(target_x, target_z)
        if pred_str == labels_str:
            exact_match += 1
        else:
            if re.match(re_str, gt_re):
                re_match += 1
            else:
                approx_match += possible_match(preds[idx], target_x, target_z)

    return re_match, approx_match

def move_along_dir(curr_x, curr_z, rot, step):
    if rot == 0:
        target_x = curr_x
        target_z = curr_z + 0.25 * step
    elif rot == 90:
        target_x = curr_x+ 0.25 * step
        target_z = curr_z
    elif rot == 180:
        target_x = curr_x
        target_z = curr_z - 0.25 * step
    elif rot == 270:
        target_x = curr_x - 0.25 * step
        target_z = curr_z
    return target_x, target_z

def possible_move(curr_queue, curr_rot, move_lower, move_upper):
    curr_q_len = len(curr_queue)
    for i in range(curr_q_len):
        curr_x, curr_z = curr_queue.popleft()
        for j in range(move_lower,move_upper+1):
            new_x, new_z = move_along_dir(curr_x, curr_z, curr_rot, j)
            if ([new_x, new_z]) not in curr_queue:
                curr_queue.append([new_x, new_z])

    return curr_queue

def possible_match(pred_traj, target_x, target_z, print_option=False):
    curr_rot = 0
    possible_queue = deque([[0.0,0.0]])
    for action in pred_traj:
        if len(possible_queue)> 1000:
            return 0
        if action == 3:
            possible_queue = possible_move(possible_queue, curr_rot, 1, 10)
        if action == 4:
            curr_rot -= 90
            curr_rot %= 360
        if action == 5:
            curr_rot += 90
            curr_rot %= 360
        if action == 6:
            curr_rot += 180
            curr_rot %= 360
        if action == 9:
            curr_rot += 180
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 1, 5)
        if action == 10:
            curr_rot -= 90
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 0, 2)
        if action == 11:
            curr_rot += 90
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 0, 2)
        if action == 7:
            possible_queue = possible_move(possible_queue, curr_rot, 0, 6)
            curr_rot -= 90
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 1, 2)
            curr_rot += 90
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 0, 6)
        if action == 8:
            possible_queue = possible_move(possible_queue, curr_rot, 0, 6)
            curr_rot += 90
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 1, 2)
            curr_rot -= 90
            curr_rot %= 360
            possible_queue = possible_move(possible_queue, curr_rot, 0, 6)
    if print_option:
        print(possible_queue)
    if [target_x, target_z] in possible_queue:
        # print("Approx!")
        return 1
    else:
        # print("Fail!")
        return 0
