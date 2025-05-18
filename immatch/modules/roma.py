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
        if args["ckpt"] != None:
            self.model = roma_outdoor(self.device,weights=torch.load(args["ckpt"])['model'], coarse_res=[672,672])
        else:
            # origin
            self.model = roma_outdoor(self.device, coarse_res=[672,672])
        self.name = f"RoMa"
        print(f"Initialize {self.name}")

    def match_pairs(self, im1_path, im2_path):
        W1, H1 = Image.open(im1_path).size
        W2, H2 = Image.open(im2_path).size

        matches, certainty = self.model.match(
            im1_path,
            im2_path,
            device=self.device
        )

        matches, certainty = self.model.sample(matches, certainty, num=5000)
        kpts1, kpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)

        kpts1[:, 0] = torch.clamp(kpts1[:, 0], 0, W1 - 1)
        kpts1[:, 1] = torch.clamp(kpts1[:, 1], 0, H1 - 1)
        kpts2[:, 0] = torch.clamp(kpts2[:, 0], 0, W2 - 1)
        kpts2[:, 1] = torch.clamp(kpts2[:, 1], 0, H2 - 1)
        matches = torch.cat((kpts1, kpts2), dim=1)

        matches = matches.cpu().numpy()

        return matches, None, None, None
