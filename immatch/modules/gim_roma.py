import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

from .base import Matching
sys.path.append(Path(__file__).parent / "../../third_party/")
from third_party.gimroma.gimroma import RoMa
from immatch.utils.data_io import load_im_tensor
import warnings
import torchvision.transforms.functional as F
import cv2

class GIMRoMa(Matching):
    def __init__(self, args):
        super().__init__()
        # raise NotImplementedError("RoMa还有问题，得到的kpts超过了图像范围，可能需要过滤一下？")

        self.model = RoMa(img_size=[672])
        self.device = "cuda"

        state_dict = torch.load(args["checkpoints_path"], map_location='cpu')
        if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)

        self.model = self.model.eval().to(self.device)

        
        # self.model = roma_outdoor(self.device)
        self.name = f"GIMRoMa"
        print(f"Initialize {self.name}")
    
    def get_padding_size(self, image, h, w):

        orig_width = image.shape[3]
        orig_height = image.shape[2]
        aspect_ratio = w / h

        new_width = max(orig_width, int(orig_height * aspect_ratio))
        new_height = max(orig_height, int(orig_width / aspect_ratio))

        pad_height = new_height - orig_height
        pad_width = new_width - orig_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        return orig_width, orig_height, pad_left, pad_right, pad_top, pad_bottom
    
    def read_image(self, path, grayscale=False):
        if grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(path), mode)
        if image is None:
            raise ValueError(f'Cannot read image {path}.')
        if not grayscale and len(image.shape) == 3:
            image = image[:, :, ::-1]  # BGR to RGB
        return image
    
    def preprocess(self, image: np.ndarray, grayscale: bool = False, resize_max: int = None,
               dfactor: int = 8):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])

        if grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        size_new = tuple(map(
                lambda x: int(x // dfactor * dfactor),
                image.shape[-2:]))
        image = F.resize(image, size=size_new)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    def match_pairs(self, im1_path, im2_path):
        image0= self.read_image(im1_path)
        image1 = self.read_image(im2_path)

        image0, _ = self.preprocess(image0)
        image1, _ = self.preprocess(image1)

        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]

        b_ids, mconf, kpts0, kpts1 = None, None, None, None
    
        width, height = 672, 672
        orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = self.get_padding_size(image0, width, height)
        orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = self.get_padding_size(image1, width, height)
        image0_ = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
        image1_ = torch.nn.functional.pad(image1, (pad_left1, pad_right1, pad_top1, pad_bottom1))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense_matches, dense_certainty = self.model.match(image0_, image1_)
            sparse_matches, mconf = self.model.sample(dense_matches, dense_certainty, 5000)

        height0, width0 = image0_.shape[-2:]
        height1, width1 = image1_.shape[-2:]

        kpts0 = sparse_matches[:, :2]
        kpts0 = torch.stack((
            width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
        kpts1 = sparse_matches[:, 2:]
        kpts1 = torch.stack((
            width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)
        b_ids = torch.where(mconf[None])[0]

        # before padding
        kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
        kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
        mask_ = (kpts0[:, 0] > 0) & \
               (kpts0[:, 1] > 0) & \
               (kpts1[:, 0] > 0) & \
               (kpts1[:, 1] > 0)
        mask_ = mask_ & \
               (kpts0[:, 0] <= (orig_width0 - 1)) & \
               (kpts1[:, 0] <= (orig_width1 - 1)) & \
               (kpts0[:, 1] <= (orig_height0 - 1)) & \
               (kpts1[:, 1] <= (orig_height1 - 1))

        mconf = mconf[mask_]
        b_ids = b_ids[mask_]
        kpts0 = kpts0[mask_]
        kpts1 = kpts1[mask_]

        # matches, certainty = self.model.match(
        #     im1_path,
        #     im2_path,
        #     device=self.device
        # )

        # matches, certainty = self.model.sample(matches, certainty, num=5000)
        # kpts1, kpts2 = self.model.to_pixel_coordinates(matches, H1, W1, H2, W2)

        # kpts1[:, 0] = torch.clamp(kpts1[:, 0], 0, W1 - 1)
        # kpts1[:, 1] = torch.clamp(kpts1[:, 1], 0, H1 - 1)
        # kpts2[:, 0] = torch.clamp(kpts2[:, 0], 0, W2 - 1)
        # kpts2[:, 1] = torch.clamp(kpts2[:, 1], 0, H2 - 1)
        matches = torch.cat((kpts0, kpts1), dim=1)

        matches = matches.cpu().numpy()

        return matches, None, None, None
