import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

from .base import Matching
sys.path.append(Path(__file__).parent / "../../third_party/")
from third_party.roma.romatch import roma_outdoor
from immatch.utils.data_io import load_im_tensor



class RoMa(Matching):
    def __init__(self, args):
        super().__init__()
        # raise NotImplementedError("RoMa还有问题，得到的kpts超过了图像范围，可能需要过滤一下？")

        self.model = roma_outdoor(self.device)
        self.name = f"RoMa"
        print(f"Initialize {self.name}")

    # def load_im(self, im_path):
    #     return load_im_tensor(
    #         im_path=im_path,
    #         device=self.device,
    #         imsize=self.imsize,
    #         normalize=False,
    #     )

    def match_pairs(self, im1_path, im2_path):
        W1, H1 = Image.open(im1_path).size
        W2, H2 = Image.open(im2_path).size

        matches, certainty = self.model.match(
            im1_path,
            im2_path,
            device=self.device
        )

<<<<<<< HEAD
        matches, certainty = self.model.sample(matches, certainty, num=5000)
        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)

        kpts1[:, 0] = torch.clamp(kpts1[:, 0], 0, W1 - 1)
        kpts1[:, 1] = torch.clamp(kpts1[:, 1], 0, H1 - 1)
        kpts2[:, 0] = torch.clamp(kpts2[:, 0], 0, W2 - 1)
        kpts2[:, 1] = torch.clamp(kpts2[:, 1], 0, H2 - 1)
        matches = torch.cat((kpts1, kpts2), dim=1)

        matches = matches.cpu().numpy()
=======
        matches, certainty = self.model.sample(warp, certainty)
        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)

        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()
        certainty = certainty.cpu().numpy()
        matches = np.concatenate((kpts1, kpts2), axis=1)
>>>>>>> 1efaf46 (mast3r,dust3r,vggt:added new models)

        return matches, None, None, None
