import sys
import numpy as np
from pathlib import Path
import os
import torchvision.transforms as tfm
from PIL import Image
import torch
import py3_wget
from typing import Union

from matplotlib import pyplot as plt

from immatch.utils.util import add_to_path, resize_to_divisible,to_normalized_coords, to_px_coords, to_numpy
from immatch.utils.geometry import geotrf
from .base import Matching

add_to_path(Path(__file__).parent / "../../third_party/mast3r")

from dust3r.inference import inference
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

WEIGHTS_DIR = Path(__file__).parent / "../../pretrained/mast3r"
WEIGHTS_DIR.mkdir(exist_ok=True)

ratios_resolutions = {
    224: {1.0: [224, 224]},
    512: {4 / 3: [512, 384], 32 / 21: [512, 336], 16 / 9: [512, 288], 2 / 1: [512, 256], 16 / 5: [512, 160]}
}

class Mast3rMatcher(Matching):
    # model_path = WEIGHTS_DIR.joinpath("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    model_path = WEIGHTS_DIR.joinpath("checkpoint-aerial-mast3r.pth")
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

    def preprocess(self, path, maxdim=512):

        # if isinstance(resize, int):
        #     resize = (resize, resize)

        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        _, H, W = img.shape
        resize_func, _, to_orig = self.get_resize_function(maxdim, self.vit_patch_size, H, W)
        

        # img_resize = resize_func(self.normalize(img))
        img_resize = resize_func(img)

        return img_resize.unsqueeze(0), to_orig
    
    def get_HW_resolution(self, H, W, maxdim, patchsize=16):
        assert maxdim in ratios_resolutions, "Error, maxdim can only be 224 or 512 for now. Other maxdims not implemented yet."
        ratios_resolutions_maxdim = ratios_resolutions[maxdim]
        mindims = set([min(res) for res in ratios_resolutions_maxdim.values()])
        ratio = W / H
        ref_ratios = np.array([*(ratios_resolutions_maxdim.keys())])
        islandscape = (W >= H)
        if islandscape:
            diff = np.abs(ratio - ref_ratios)
        else:
            diff = np.abs(ratio - (1 / ref_ratios))
        selkey = ref_ratios[np.argmin(diff)]
        res = ratios_resolutions_maxdim[selkey]
        # check patchsize and make sure output resolution is a multiple of patchsize
        if isinstance(patchsize, tuple):
            assert len(patchsize) == 2 and isinstance(patchsize[0], int) and isinstance(
                patchsize[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patchsize[0] == patchsize[1], "Error, non square patches not managed"
            patchsize = patchsize[0]
        assert max(res) == maxdim
        assert min(res) in mindims
        return res[::-1] if islandscape else res  # return HW


    def get_resize_function(self, maxdim, patch_size, H, W, is_mask=False):
        if [max(H, W), min(H, W)] in ratios_resolutions[maxdim].values():
            return lambda x: x, np.eye(3), np.eye(3)
        else:
            target_HW = self.get_HW_resolution(H, W, maxdim=maxdim, patchsize=patch_size)

            ratio = W / H
            target_ratio = target_HW[1] / target_HW[0]
            to_orig_crop = np.eye(3)
            to_rescaled_crop = np.eye(3)
            if abs(ratio - target_ratio) < np.finfo(np.float32).eps:
                crop_W = W
                crop_H = H
            elif ratio - target_ratio < 0:
                crop_W = W
                crop_H = int(W / target_ratio)
                to_orig_crop[1, 2] = (H - crop_H) / 2.0
                to_rescaled_crop[1, 2] = -(H - crop_H) / 2.0
            else:
                crop_W = int(H * target_ratio)
                crop_H = H
                to_orig_crop[0, 2] = (W - crop_W) / 2.0
                to_rescaled_crop[0, 2] = - (W - crop_W) / 2.0

            crop_op = tfm.CenterCrop([crop_H, crop_W])

            if is_mask:
                resize_op = tfm.Resize(size=target_HW, interpolation=tfm.InterpolationMode.NEAREST_EXACT)
            else:
                resize_op = tfm.Resize(size=target_HW)
            to_orig_resize = np.array([[crop_W / target_HW[1], 0, 0],
                                    [0, crop_H / target_HW[0], 0],
                                    [0, 0, 1]])
            to_rescaled_resize = np.array([[target_HW[1] / crop_W, 0, 0],
                                        [0, target_HW[0] / crop_H, 0],
                                        [0, 0, 1]])

            op = tfm.Compose([crop_op, resize_op])

            return op, to_rescaled_resize @ to_rescaled_crop, to_orig_crop @ to_orig_resize
    
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
        img0, scale0 = self.preprocess(img0_path)
        img1, scale1= self.preprocess(img1_path)

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
        # from immatch.utils.visualize import plot_match_2view
        # plot_match_2view(img0, img1, mkpts0, mkpts1, "1.jpg", percent=1)
        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        # H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = geotrf(scale0, mkpts0, norm=True)
        mkpts1 = geotrf(scale1, mkpts1, norm=True)
        matches = np.concatenate([mkpts0, mkpts1], axis=1)

        return matches, mkpts0, mkpts1, len(mkpts0)