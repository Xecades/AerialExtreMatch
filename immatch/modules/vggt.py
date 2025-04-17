import sys
import numpy as np
from pathlib import Path
import os
import torchvision.transforms as tfm
from PIL import Image
import torch
import cv2
import py3_wget
from typing import Union

from immatch.utils.util import add_to_path, to_normalized_coords, to_px_coords
from .base import Matching

add_to_path(Path(__file__).parent / "../../third_party/vggt")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.ALIKED.nets.aliked import ALIKED

WEIGHTS_DIR = Path(__file__).parent / "../../pretrained/vggt"
WEIGHTS_DIR.mkdir(exist_ok=True)

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

class VGGTMatcher(Matching):
    model_path = WEIGHTS_DIR.joinpath("model.pt")
    vit_patch_size = 16

    def __init__(self,*args):
        super().__init__()
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.verbose = False

        self.download_weights()
        model = VGGT()
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        self.vggt_model = model.to('cuda')
        self.extract_model = ALIKED(model_name="aliked-n16rot",
                  device='cuda',
                  top_k=-1,
                  scores_th=0.2,
                  n_limit=2000)

    @staticmethod
    def download_weights():
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        if not os.path.isfile(VGGTMatcher.model_path):
            print("Downloading VGGT... (takes a while)")
            py3_wget.download_file(url, VGGTMatcher.model_path)

    def preprocess(self, path0, path1):

        original_images = []
        img_paths = [path0, path1]
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            original_images.append(np.array(img))
        
        images = load_and_preprocess_images(img_paths).to('cuda')
        # print(f"Preprocessed images shape: {images.shape}")

        S, C, H, W, = images.shape
        normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
        
        for i, img in enumerate(original_images):
            resized_img = cv2.resize(img, (W, H))
            normalized_images[i] = resized_img / 255.0
        
        return images, normalized_images, original_images
    
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
        images,  normalized_images, original_images = self.preprocess(img0_path, img1_path)
        init_image = images[0,:,:,:].unsqueeze(0)
        predict_init = self.extract_model.run(init_image)

        query_points = torch.FloatTensor(predict_init["keypoints"]).to('cuda')

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]
                aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(images)
                track_list, vis_score, conf_score = self.vggt_model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

        matches_im0 = track_list[1].squeeze(0)[0,:,:].cpu().numpy()
        matches_im1 = track_list[1].squeeze(0)[1,:,:].cpu().numpy()

        ori_H0, ori_W0 = original_images[0].shape[:2]
        ori_H1, ori_W1 = original_images[1].shape[:2]
        H0, W0 = images[0,0].shape[1:]
        H1, W1 = images[0,1].shape[1:]

        mkpts0 = self.rescale_coords(matches_im0, ori_H0, ori_W0, H0, W0)
        mkpts1 = self.rescale_coords(matches_im1, ori_H1, ori_W1, H1, W1)
        matches = np.concatenate([mkpts0, mkpts1], axis=1)

        return matches, mkpts0, mkpts1, len(mkpts0)
