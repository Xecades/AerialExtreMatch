import torch
import numpy as np

from .base import Matching
from third_party.LightGlue.lightglue import LightGlue, ALIKED
from immatch.utils.data_io import load_im_tensor


class ALIKED_LightGlue(Matching):
    def __init__(self, args):
        super().__init__()
        self.imsize = args["imsize"]
        self.no_match_upscale = args["no_match_upscale"]

        self.extractor = ALIKED().eval().to(self.device)
        self.matcher = LightGlue(features="aliked").eval().to(self.device)

        self.name = f"ALIKED_LightGlue"
        if self.no_match_upscale:
            self.name += "_noms"

        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_im_tensor(
            im_path=im_path,
            device=self.device,
            imsize=self.imsize,
        )

    def match_inputs_(self, im1: torch.Tensor, im2: torch.Tensor):
        # print(im1.shape)  # [1, 3, 899, 769]
        # print(im2.shape)  # [1, 3, 1024, 768]

        feat1 = self.extractor.extract(im1)
        feat2 = self.extractor.extract(im2)

        kpts1, kpts2 = feat1["keypoints"][0], feat2["keypoints"][0]

        out = self.matcher({'image0': feat1, 'image1': feat2})

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
