import numpy as np
import torchvision.transforms.functional as tvf
import torch
import os
from PIL import Image

class ExtreData:
    def __init__(
        self,
        data_root: str,
        scene_info: dict,
    ) -> None:
        self.data_root = data_root
        self.image_paths = scene_info["rgb"]
        self.depth_paths = scene_info["depth"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]

        self.pairs = scene_info["pair"]
        self.overlaps = np.array(scene_info["overlap"])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        K1 = torch.tensor(self.intrinsics[idx1].copy()).float().reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy()).float().reshape(3, 3)

        # read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1))).float()[:4, :4]

        # Load positive pair data
        im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        im_A_ref = os.path.join(self.data_root, im_A)
        im_B_ref = os.path.join(self.data_root, im_B)
        depth_A_ref = os.path.join(self.data_root, depth1)
        depth_B_ref = os.path.join(self.data_root, depth2)

        data_dict = {
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
            "depth_A_path":depth_A_ref,
            "depth_B_path":depth_B_ref
        }

        return data_dict