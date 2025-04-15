import numpy as np
import torch
import os


class MegaDepthData:
    def __init__(
        self,
        data_root: str,
        scene_info: dict,
    ) -> None:
        self.data_root = data_root
        self.image_paths = scene_info["image_paths"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]
        self.pair_infos = scene_info["pair_infos"]

    def __len__(self):
        return len(self.pair_infos)

    def __getitem__(self, pair_idx):
        # Get pair indices and overlap information
        (idx1, idx2), overlap, _ = self.pair_infos[pair_idx]

        # Read intrinsics of original size
        K1 = torch.tensor(self.intrinsics[idx1].copy()).float().reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy()).float().reshape(3, 3)

        # Read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1))).float()[:4, :4]

        # Load image paths
        im_A = self.image_paths[idx1].replace("Undistorted_SfM/", "")
        im_B = self.image_paths[idx2].replace("Undistorted_SfM/", "")
        im_A_ref = os.path.join(self.data_root, im_A)
        im_B_ref = os.path.join(self.data_root, im_B)

        data_dict = {
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
            "overlap": overlap
        }

        return data_dict


class MegaDepthBuilder:
    def __init__(self, data_root: str = "./data/datasets/MegaDepth_undistort") -> None:
        self.data_root = data_root

    def build_from_npz_list(self, npz_list_path, npz_root):
        with open(npz_list_path, "r") as f:
            npz_names = [name.split()[0] for name in f.readlines()]

        scenes = []
        for name in npz_names:
            scene_info_path = os.path.join(npz_root, f"{name}.npz")
            scene_info = np.load(scene_info_path, allow_pickle=True)
            scenes.append(
                MegaDepthData(
                    data_root=self.data_root,
                    scene_info=scene_info
                )
            )
        return scenes
