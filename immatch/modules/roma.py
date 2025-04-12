import torch
import numpy as np
from PIL import Image

from .base import Matching
from third_party.roma.romatch import roma_outdoor
from immatch.utils.data_io import load_im_tensor


class RoMa(Matching):
    def __init__(self, args):
        super().__init__()
        raise NotImplementedError("RoMa还有问题，得到的kpts超过了图像范围，可能需要过滤一下？")

        self.model = roma_outdoor(self.device)
        self.name = f"RoMa"
        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_im_tensor(
            im_path=im_path,
            device=self.device,
            imsize=self.imsize,
            normalize=False,
        )

    def match_pairs(self, im1_path, im2_path):
        W1, H1 = Image.open(im1_path).size
        W2, H2 = Image.open(im2_path).size

        warp, certainty = self.model.match(
            im1_path,
            im2_path,
            device=self.device
        )

        matches, certainty = self.model.sample(warp, certainty, num=2048)
        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)

        # kpts1 = torch.cla
        q = kpts1.round().long()
        print(q.shape)
        print(W1, H1, W2, H2)
        print(torch.max(q), torch.min(q))

        kpts1 = kpts1.cpu().numpy()
        kpts2 = kpts2.cpu().numpy()
        certainty = certainty.cpu().numpy()
        matches = np.concatenate((kpts1, kpts2), axis=1)

        return matches, kpts1, kpts2, certainty
