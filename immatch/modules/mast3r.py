import sys
import numpy as np
from pathlib import Path
import os
import torchvision.transforms as tfm
from PIL import Image
import torch
import py3_wget
from typing import Union

from immatch.utils.util import add_to_path, resize_to_divisible,to_normalized_coords, to_px_coords, to_numpy
from .base import Matching

add_to_path(Path(__file__).parent / "../../third_party/mast3r")

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

WEIGHTS_DIR = Path(__file__).parent / "../../pretrained/mast3r"
WEIGHTS_DIR.mkdir(exist_ok=True)

class Mast3rMatcher(Matching):
    model_path = WEIGHTS_DIR.joinpath("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    vit_patch_size = 16

    def __init__(self,*args):
        super().__init__()
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.verbose = False

        self.download_weights()
        self.model = AsymmetricMASt3R.from_pretrained(self.model_path).to('cuda')

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if not os.path.isfile(Mast3rMatcher.model_path):
            print("Downloading Mast3r(ViT large)... (takes a while)")
            py3_wget.download_file(url, Mast3rMatcher.model_path)

    def preprocess(self, path, resize=512):

        if isinstance(resize, int):
            resize = (resize, resize)

        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        if resize is not None:
            img_resize = tfm.Resize(resize, antialias=True)(img)
        _, h, w = img.shape
        orig_shape = h, w

        img_resize = resize_to_divisible(img_resize, self.vit_patch_size)

        img_resize = self.normalize(img_resize).unsqueeze(0)

        return img_resize, orig_shape
    
    def rescale_coords(
        self,
        pts: Union[np.ndarray, torch.Tensor],
        h_orig: int,
        w_orig: int,
        h_new: int,
        w_new: int,
    ) -> np.ndarray:
        """Rescale kpts coordinates from one img size to another

        Args:
            pts (np.ndarray | torch.Tensor): (N,2) array of kpts
            h_orig (int): height of original img
            w_orig (int): width of original img
            h_new (int): height of new img
            w_new (int): width of new img

        Returns:
            np.ndarray: (N,2) array of kpts in original img coordinates
        """
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    def match_pairs(self, img0_path, img1_path):
        img0, img0_orig_shape = self.preprocess(img0_path)
        img1, img1_orig_shape = self.preprocess(img1_path)

        img_pair = [
            {"img": img0, "idx": 0, "instance": 0, "true_shape": np.int32([img0.shape[-2:]])},
            {"img": img1, "idx": 1, "instance": 1, "true_shape": np.int32([img1.shape[-2:]])},
        ]
        output = inference([tuple(img_pair)], self.model, self.device, batch_size=1, verbose=False)
        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = pred1["desc"].squeeze(0).detach(), pred2["desc"].squeeze(0).detach()

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2, subsample_or_initxy1=8, device=self.device, dist="dot", block_size=2**13
        )

        # ignore small border around the edge
        H0, W0 = view1["true_shape"][0]
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3)
            & (matches_im0[:, 0] < int(W0) - 3)
            & (matches_im0[:, 1] >= 3)
            & (matches_im0[:, 1] < int(H0) - 3)
        )

        H1, W1 = view2["true_shape"][0]
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3)
            & (matches_im1[:, 0] < int(W1) - 3)
            & (matches_im1[:, 1] >= 3)
            & (matches_im1[:, 1] < int(H1) - 3)
        )

        valid_matches = valid_matches_im0 & valid_matches_im1
        mkpts0, mkpts1 = matches_im0[valid_matches], matches_im1[valid_matches]
        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)
        matches = np.concatenate([mkpts0, mkpts1], axis=1)

        return matches, mkpts0, mkpts1, len(mkpts0)