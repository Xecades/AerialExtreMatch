import torch
import numpy as np

from .base import Matching
import kornia as K
import kornia.feature as KF
from immatch.utils.data_io import load_im_tensor


class DISK_LightGlue(Matching):
    def __init__(self, args):
        super().__init__()
        self.imsize = args["imsize"]
        self.num_features = args["num_features"]
        self.no_match_upscale = args["no_match_upscale"]

        self.device = K.utils.get_cuda_or_mps_device_if_available()

        self.disk = KF.DISK.from_pretrained("depth").eval().to(self.device)
        self.lg = KF.LightGlue("disk").eval().to(self.device)

        self.name = f"DISK_LightGlue"
        if self.no_match_upscale:
            self.name += "_noms"

        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_im_tensor(
            im_path=im_path,
            device=self.device,
            imsize=self.imsize,
        )
        # return load_gray_scale_tensor_cv(im_path, self.device, imsize=self.imsize)
        # return K.io.load_image(im_path, K.io.ImageLoadType.RGB32, device=self.device)[None, ...]

    def match_inputs_(self, im1: torch.Tensor, im2: torch.Tensor):
        # print(im1.shape)  # [1, 3, 899, 769]
        # print(im2.shape)  # [1, 3, 1024, 768]

        feat1 = self.disk(im1, self.num_features, pad_if_not_divisible=True)[0]
        feat2 = self.disk(im2, self.num_features, pad_if_not_divisible=True)[0]

        kpts1, descs1 = feat1.keypoints, feat1.descriptors
        kpts2, descs2 = feat2.keypoints, feat2.descriptors

        # print(kpts1.shape, descs1.shape)
        # print(kpts2.shape, descs2.shape)
        # torch.Size([2048, 2]) torch.Size([2048, 128])
        # torch.Size([2048, 2]) torch.Size([2048, 128])

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

        out = self.lg({"image0": image0, "image1": image1})
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
        im1, sc1 = self.load_im(im1_path)
        im2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(im1, im2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2
        return matches, kpts1, kpts2, scores
