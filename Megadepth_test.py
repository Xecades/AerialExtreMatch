import immatch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import torch
import numpy as np
from immatch.datasets.megadepth_dataset import MegaDepthBuilder
from immatch.utils.metrics import cal_relapose_error, cal_relapose_auc

if __name__ == "__main__":
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
model_name = "eloftr"

# Initialize model
with open(f"configs/{model_name}.yml", "r") as f:
    cfgs = yaml.load(f, Loader=yaml.FullLoader)
    args = cfgs["megadepth"] if "megadepth" in cfgs else cfgs["example"]
model = immatch.__dict__[args["class"]](args)
def matcher(im1, im2): return model.match_pairs(im1, im2)


# Data loading
megadepth_builder = MegaDepthBuilder(data_root="./data/datasets/MegaDepth_undistort",
                                     npz_root="./third_party/aspanformer/assets/megadepth_test_1500_scene_info")
name = "megadepth"
save_path = "temp/megadepth/" + name + "_" + model_name + ".txt"
test_scenes = megadepth_builder.build_from_npz_list(
    "./third_party/aspanformer/assets/megadepth_test_1500_scene_info/megadepth_test_1500.txt")

# Statistics for storing results
total_auc_1 = 0.0
total_auc_3 = 0.0
total_auc_5 = 0.0
total_auc_10 = 0.0
total_auc_20 = 0.0
scene_count = 0

for scene_data in test_scenes:
    sampler = RandomSampler(scene_data, num_samples=50)
    val_loader = DataLoader(scene_data, sampler=sampler, num_workers=0,
                            batch_size=1, pin_memory=True)
    scene_count += 1

    # For storing pose errors in this scene
    statis = {
        "R_errs": [],
        "t_errs": [],
        "inliers": [],
        "failed": []
    }

    for batch_idx, batch in tqdm(enumerate(val_loader)):
        # Get image paths and camera parameters
        im1, im2 = batch["im_A_path"][0], batch["im_B_path"][0]
        K1 = batch["K1"].to(device)
        K2 = batch["K2"].to(device)
        T_1to2 = batch["T_1to2"].to(device)

        # Extract features and match
        matches, pts1, pts2, _ = matcher(im1, im2)

        # If insufficient matches, record failure
        if len(pts1) < 5:
            statis["failed"].append(len(statis["R_errs"]))
            statis["R_errs"].append(np.inf)
            statis["t_errs"].append(np.inf)
            statis["inliers"].append(0)
            continue

        # Calculate essential matrix and decompose to get relative pose
        pts1_np = pts1.copy()
        pts2_np = pts2.copy()
        K1_np = K1.cpu().numpy()[0]
        K2_np = K2.cpu().numpy()[0]

        # Normalize keypoints
        pts1_norm = (pts1_np - K1_np[[0, 1], [2, 2]]
                     [None]) / K1_np[[0, 1], [0, 1]][None]
        pts2_norm = (pts2_np - K2_np[[0, 1], [2, 2]]
                     [None]) / K2_np[[0, 1], [0, 1]][None]

        # Calculate essential matrix and recover pose
        ransac_thres = 0.5 / \
            np.mean([K1_np[0, 0], K1_np[1, 1], K2_np[0, 0], K2_np[1, 1]])
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm, np.eye(3),
            threshold=ransac_thres, prob=0.99999, method=cv2.RANSAC
        )

        if E is None:
            statis["failed"].append(len(statis["R_errs"]))
            statis["R_errs"].append(np.inf)
            statis["t_errs"].append(np.inf)
            statis["inliers"].append(0)
            continue

        # Recover pose from essential matrix
        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) // 3):
            n, R, t, _ = cv2.recoverPose(
                _E, pts1_norm, pts2_norm, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                ret = (R, t[:, 0], mask.ravel() > 0)
                best_num_inliers = n

        if ret is None:
            statis["failed"].append(len(statis["R_errs"]))
            statis["R_errs"].append(np.inf)
            statis["t_errs"].append(np.inf)
            statis["inliers"].append(0)
            continue

        R, t, inliers = ret

        # Get ground truth relative pose
        T_gt = T_1to2[0].cpu().numpy()
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]

        # Calculate pose errors
        R_err, t_err = cal_relapose_error(R, R_gt, t, t_gt)

        # Record results
        statis["R_errs"].append(R_err)
        statis["t_errs"].append(t_err)
        statis["inliers"].append(inliers.sum() / len(pts1))

        # Optional: save matching visualization
        # immatch.utils.plot_matches(im1, im2, matches, radius=2, lines=True, sav_fig=f"matches_{model_name}_megadepth.png")

    # Evaluate results for this scene
    thresholds = [1, 3, 5, 10, 20]
    pose_auc = cal_relapose_auc(statis, thresholds=thresholds)

    # Accumulate AUC scores
    total_auc_1 += pose_auc[0]
    total_auc_3 += pose_auc[1]
    total_auc_5 += pose_auc[2]
    total_auc_10 += pose_auc[3]
    total_auc_20 += pose_auc[4]

# Calculate average AUC across all scenes
result = {
    "auc_1": total_auc_1 / scene_count,
    "auc_3": total_auc_3 / scene_count,
    "auc_5": total_auc_5 / scene_count,
    "auc_10": total_auc_10 / scene_count,
    "auc_20": total_auc_20 / scene_count
}

with open(save_path, "w") as f_w:
    for key in result.keys():
        info = key + " " + str(result[key]) + "\n"
        f_w.write(info)
f_w.close()
