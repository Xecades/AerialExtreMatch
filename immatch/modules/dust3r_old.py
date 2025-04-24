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

add_to_path(Path(__file__).parent / "../../third_party/dust3r")

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

WEIGHTS_DIR = Path(__file__).parent / "../../pretrained/dust3r"
WEIGHTS_DIR.mkdir(exist_ok=True)

class Dust3rMatcher(Matching):
    model_path = WEIGHTS_DIR.joinpath("duster_vit_large.pth")
    vit_patch_size = 16

    def __init__(self,*args):
        super().__init__()
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.verbose = False

        self.download_weights()
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path).to('cuda')

    @staticmethod
    def download_weights():
        url = "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"

        if not os.path.isfile(Dust3rMatcher.model_path):
            print("Downloading Dust3r(ViT large)... (takes a while)")
            py3_wget.download_file(url, Dust3rMatcher.model_path)

    def preprocess(self, path, resize=512):

        # if isinstance(resize, int):
            # resize = (resize, resize)

        img = tfm.ToTensor()(Image.open(path).convert("RGB"))
        _, h, w = img.shape
        orig_shape = h, w
        if resize is not None:
            img_resize = tfm.Resize(resize, antialias=True)(img)
            # if h > w:
            #     new_h = resize
            #     new_w = int(w* (resize / h))
            # else:
            #     new_w = resize
            #     new_h = int(h * (resize / w))
        

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

        images = [
            {"img": img0, "idx": 0, "instance": 0},
            {"img": img1, "idx": 1, "instance": 1},
        ]
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(pairs, self.model, self.device, batch_size=1, verbose=self.verbose)

        scene = global_aligner(
            output,
            device=self.device,
            mode=GlobalAlignerMode.PairViewer,
            verbose=self.verbose,
        )
        # retrieve useful values from scene:
        confidence_masks = scene.get_masks()
        pts3d = scene.get_pts3d()
        imgs = scene.imgs
        pts2d_list, pts3d_list = [], []

        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
            
        # return if there is no 3d points found on either one of the image
        if pts3d_list[0].shape[0] == 0 or pts3d_list[1].shape[0] == 0:
            return np.empty((0,2)), np.empty((0,2)), None, None
        reciprocal_in_P2, nn2_in_P1, _ = find_reciprocal_matches(*pts3d_list)

        mkpts1 = pts2d_list[1][reciprocal_in_P2]
        mkpts0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # duster sometimes requires reshaping an image to fit vit patch size evenly, so we need to
        # rescale kpts to the original img
        H0, W0, H1, W1 = *img0.shape[-2:], *img1.shape[-2:]
        mkpts0 = self.rescale_coords(mkpts0, *img0_orig_shape, H0, W0)
        mkpts1 = self.rescale_coords(mkpts1, *img1_orig_shape, H1, W1)
        matches = np.concatenate([mkpts0, mkpts1], axis=1)

        return matches, mkpts0, mkpts1, len(mkpts0)