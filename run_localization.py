import warnings
warnings.filterwarnings('ignore')
import torch
import yaml
import os
from tqdm import tqdm

import immatch
from immatch.utils.crop4map import generate_ref_map
from immatch.utils.transform import *
from immatch.utils.visualize import plot_match
from immatch.utils.util import aggregate_stats
from immatch.localize.dom_localize import DOMLocalizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_path = "/media/guan/data/sparse/ExtreBenchmark/localization_config/config_roma_syn.yml"
config = load_config(config_path)

# setting initialization
data_path = config["data_path"]
dataset = config["dataname"]
# 是否使用yaw角先验
yaw_flag = config["yaw_flag"]
# query
query_config = config["query"]
query_path = query_config["query_path"]
query_img_path = query_config["query_img_path"]
query_prior_path = query_config["query_prior_path"]
query_intrinsic_path = query_config["query_intrinsic_path"]
query_rot_img_path = query_config["query_rot_img_path"]
ref_save = query_config["ref_save"]
if not os.path.exists(query_rot_img_path):
    os.makedirs(query_rot_img_path)
# ref
ref_config = config["ref"]
ref_DSM_path = ref_config["ref_DSM_path"]
ref_DOM_path = ref_config["ref_DOM_path"]
ref_npy_path = ref_config["ref_npy_path"]

# matching init
result_config = config["result"]
matching_path = os.path.join(data_path, result_config["matching_path_name"])
if not os.path.exists(matching_path):
    os.mkdir(matching_path)

result_txt = data_path + '/' + result_config["result_txt"]

# matching method
matching_method = config["method"]
with open(f"configs/{matching_method}.yml", "r") as f:
    args = yaml.load(f, Loader=yaml.FullLoader)["example"]
model = immatch.__dict__[args["class"]](args)
def matcher(im1, im2): return model.match_pairs(im1, im2)

# reference map && pair
query_pose = parse_pose_list(query_prior_path)
query_intrinsic = parse_intrinsic_list(query_intrinsic_path)
query_osg = parse_pose_osg(query_prior_path)
data = generate_ref_map(query_intrinsic, query_pose, ref_DSM_path, ref_DOM_path, ref_npy_path, ref_save)
# 
poses_errors = []
angle_errors = []

with open(result_txt, 'w') as f_result:
    for qname in tqdm(data.keys()):
        imgr_path = os.path.join(ref_save, 'rgb', data[qname]['imgr_name'] + '.jpg')
        exr_path = os.path.join(ref_save, 'npy', data[qname]['exr_name'] + '.npy')
        q_intrinsic = query_intrinsic[qname]
        ori_path = os.path.join(query_img_path, qname+'.jpg')
        rot_R = None
        if yaw_flag:
            yaw = query_osg[qname][0][2]
            rgb_img = cv2.imread(ori_path)
            rot_img, rot_R = rotate_image(rgb_img, yaw)
            rot_img_save_path = os.path.join(query_rot_img_path, qname+'.jpg')
            cv2.imwrite(rot_img_save_path, rot_img)
            mathes, _, _, _ = matcher(rot_img_save_path, imgr_path)
            plot_match(ori_path, rot_img_save_path, imgr_path, mathes, matching_path, qname, rot_R)
        else:
            mathes, _, _, _ = matcher(ori_path, imgr_path)
        
        
        if len(mathes) <= 4:
            continue

        domlocalizer = DOMLocalizer(
            K1=q_intrinsic,
            points_path=exr_path,
            matches=mathes
        )
        pred_matrix = domlocalizer.main_localize(ori_path, query_pose[qname], rot_R)
        # eval
        error_t, error_r = domlocalizer.eval(
                        pred_matrix=pred_matrix,
                        gt_pose=query_pose[qname],
                    )
        f_result.write(str(qname) + ' ' + str(error_t) + ' ' + str(error_r) + '\n')
        poses_errors.append(error_t)
        angle_errors.append(error_r)

out_string = aggregate_stats(f'{dataset}', poses_errors, angle_errors)
print(out_string)
f_result.write(out_string)