import torch
import numpy as np

from .base import Matching
import kornia as K
import kornia.feature as KF
from .superpoint import SuperPoint
from immatch.utils.data_io import load_gray_scale_tensor_cv


class SP_LightGlue(Matching):
    def __init__(self, args):
        super().__init__()
        self.imsize = args["imsize"]
        self.no_match_upscale = args["no_match_upscale"]

        self.device = K.utils.get_cuda_or_mps_device_if_available()

        self.model = KF.LightGlue("superpoint").eval().to(self.device)
        self.detector = SuperPoint(args)

        rad = self.detector.model.config["nms_radius"]
        self.name = f"SP_LightGlue_r{rad}"
        if self.no_match_upscale:
            self.name += "_noms"

        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_gray_scale_tensor_cv(
            im_path=im_path,
            device=self.device,
            imsize=self.imsize,
        )

    def match_inputs_(self, im1: torch.Tensor, im2: torch.Tensor):
        pred1 = self.detector.model({"image": im1})
        pred2 = self.detector.model({"image": im2})

        kpts1, descs1 = pred1["keypoints"][0], pred1["descriptors"][0]
        kpts2, descs2 = pred2["keypoints"][0], pred2["descriptors"][0]

        descs1 = descs1.permute(1, 0)
        descs2 = descs2.permute(1, 0)

        # print(kpts1.shape, descs1.shape)
        # print(kpts2.shape, descs2.shape)
        # torch.Size([377, 2]) torch.Size([377, 256])
        # torch.Size([698, 2]) torch.Size([698, 256])

        image0 = {
            "keypoints": kpts1[None],
            "descriptors": descs1[None],
            "image_size": torch.tensor(im1.shape[-2:][::-1]).view(1, 2).to(self.device),
        }
        image1 = {
            "keypoints": kpts2[None],
            "descriptors": descs2[None],
            "image_size": torch.tensor(im2.shape[-2:][::-1]).view(1, 2).to(self.device),
        }

        out = self.model({"image0": image0, "image1": image1})
        idxs = out["matches"][0]  # matches are indexes of keypoints
        scores = out["scores"][0]

        mkpts1 = kpts1[idxs[:, 0]]
        mkpts2 = kpts2[idxs[:, 1]]
        matches = torch.cat([mkpts1, mkpts2], dim=1)

        matches = matches.cpu().numpy()
        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()
        scores = scores.cpu().numpy()

        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path):
        im1, sc1, _ = self.load_im(im1_path)
        im2, sc2, _ = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(im1, im2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
