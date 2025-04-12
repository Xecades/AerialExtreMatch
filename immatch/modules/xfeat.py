import numpy as np

from .base import Matching
from immatch.utils.data_io import load_im_tensor

# Use if statement to prevent formatting from corrupting the code
if True:
    import sys
    from pathlib import Path
    xfeat_path = Path(__file__).parent / "../../third_party/xfeat"
    sys.path.append(str(xfeat_path))
    from third_party.xfeat.modules.xfeat import XFeat as _XFeat


class XFeat(Matching):
    def __init__(self, args):
        super().__init__()

        self.imsize = args.get("imsize", -1)
        self.num_keypoints = args.get("num_keypoints", 4096)

        self.model = _XFeat()

        self.name = "XFeat"
        print(f"Initialize {self.name}")

    def load_im(self, im_path):
        return load_im_tensor(
            im_path,
            device=self.device,
            imsize=self.imsize,
            normalize=False,
        )

    def match_pairs(self, im1_path, im2_path):
        im1, _ = self.load_im(im1_path)
        im2, _ = self.load_im(im2_path)

        mkpts1, mkpts2 = self.model.match_xfeat(im1, im2)
        matches = np.concatenate([mkpts1, mkpts2], axis=1)

        return matches, None, None, None
