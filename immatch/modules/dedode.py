import cv2
import torch
import numpy as np

from .base import Matching
from immatch.utils.data_io import load_im_tensor
import kornia as K
import kornia.feature as KF


class DeDoDe(Matching):
    def __init__(self, args):
        super().__init__()
        self.name = "DeDoDe"

        self.imsize = args.get("imsize", -1)
        self.match_threshold = args.get("match_threshold", 0.0)
        self.num_keypoints = args.get("num_keypoints", 4096)

        self.model = KF.DeDoDe.from_pretrained(
            detector_weights="L-upright",
            descriptor_weights="B-upright",
        ).to(self.device)

        print(f"Initialize {self.name}")

    def load_and_extract(self, im_path):
        img, scale = load_im_tensor(
            im_path, device=self.device, imsize=self.imsize, normalize=False)

        keypoints, scores, descriptors = self.model(img, self.num_keypoints)
        keypoints -= 0.5

        kpts = keypoints[0]
        desc = descriptors[0]

        return kpts, desc, scale

    def match_pairs(self, im1_path, im2_path):
        kpts1, desc1, scale1 = self.load_and_extract(im1_path)
        kpts2, desc2, scale2 = self.load_and_extract(im2_path)

        dists, idxs = KF.match_mnn(desc1, desc2)

        if idxs.shape[0] == 0:
            return np.zeros((0, 4)), kpts1.cpu().numpy(), kpts2.cpu().numpy(), np.zeros(0)

        mkpts1 = kpts1[idxs[:, 0]].cpu().numpy()
        mkpts2 = kpts2[idxs[:, 1]].cpu().numpy()
        matches = np.concatenate([mkpts1, mkpts2], axis=1)
        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()

        scores = (1.0 - dists.cpu().numpy())

        upscale = np.array([scale1 + scale2])
        matches = upscale * matches
        kpts1 = scale1 * kpts1
        kpts2 = scale2 * kpts2

        return matches, kpts1, kpts2, scores
