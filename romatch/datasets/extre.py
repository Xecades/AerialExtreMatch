import os
from PIL import Image
from romatch.utils import get_tuple_transform_ops, get_depth_tuple_transform_ops
from romatch.utils.transforms import RandomColorAug
import torchvision.transforms.functional as tvf
import numpy as np
import torch

if True:
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2


class ExtreScene:
    def __init__(
        self,
        data_root: str,
        scene_info: dict,
        scene_name=None,
        ht=560,
        wt=560,
        min_overlap=0.0,
        max_overlap=1.0,
        shake_t=0,
        normalize=True,
        use_horizontal_flip_aug=False,
        use_single_horizontal_flip_aug=False,
        random_eraser=None,
        use_randaug=False,
        max_num_pairs=20000,  # * total 2499030
    ) -> None:
        self.data_root = data_root
        self.scene_name = os.path.splitext(scene_name)[0]\
            + f"_{min_overlap}_{max_overlap}"
        self.image_paths = scene_info["rgb"]
        self.depth_paths = scene_info["depth"]
        self.intrinsics = scene_info["intrinsics"]
        self.poses = scene_info["poses"]

        self.pairs = np.array(scene_info["pair"])
        self.overlaps = np.array(scene_info["overlap"])

        thres = (self.overlaps > min_overlap) \
            & (self.overlaps < max_overlap)
        self.pairs = self.pairs[thres]
        self.overlaps = self.overlaps[thres]

        if len(self.pairs) > max_num_pairs:
            pairinds = np.random.choice(
                np.arange(0, len(self.pairs)),
                max_num_pairs,
                replace=False
            )
            self.pairs = self.pairs[pairinds]
            self.overlaps = self.overlaps[pairinds]

        self.im_transform_ops = get_tuple_transform_ops(
            resize=(ht, wt),
            normalize=normalize,
        )
        self.depth_transform_ops = get_depth_tuple_transform_ops(
            resize=(ht, wt)
        )
        self.wt, self.ht = wt, ht
        self.shake_t = shake_t

        if use_horizontal_flip_aug and use_single_horizontal_flip_aug:
            raise ValueError("Can't both flip both images and only flip one")
        self.use_horizontal_flip_aug = use_horizontal_flip_aug
        self.use_single_horizontal_flip_aug = use_single_horizontal_flip_aug

        self.use_randaug = use_randaug
        self.random_eraser = random_eraser

    def load_im(self, path):
        return Image.open(path)

    def horizontal_flip(self, im_A, im_B, depth_A, depth_B,  K_A, K_B):
        im_A = im_A.flip(-1)
        im_B = im_B.flip(-1)
        depth_A, depth_B = depth_A.flip(-1), depth_B.flip(-1)
        flip_mat = torch.tensor(
            [[-1, 0, self.wt], [0, 1, 0], [0, 0, 1.]]).to(K_A.device)
        K_A = flip_mat@K_A
        K_B = flip_mat@K_B

        return im_A, im_B, depth_A, depth_B, K_A, K_B

    def load_depth(self, depth_ref):
        depth = cv2.imread(depth_ref, cv2.IMREAD_UNCHANGED)
        return torch.tensor(depth[:, :, 0])

    def __len__(self):
        return len(self.pairs)

    def scale_intrinsic(self, K, wi, hi):
        sx, sy = self.wt / wi, self.ht / hi
        sK = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        return sK @ K

    def rand_shake(self, *things):
        t = np.random.choice(range(-self.shake_t, self.shake_t + 1), size=2)
        return [
            tvf.affine(thing, angle=0.0, translate=list(
                t), scale=1.0, shear=[0.0, 0.0])
            for thing in things
        ], t

    def rand_augment(self, im_A, im_B):
        im_A = np.array(im_A)
        im_B = np.array(im_B)
        random_color_aug = RandomColorAug()
        im_A = random_color_aug(im_A)
        im_B = random_color_aug(im_B)
        im_A = Image.fromarray(im_A)
        im_B = Image.fromarray(im_B)
        return im_A, im_B

    def __getitem__(self, pair_idx):
        # read intrinsics of original size
        idx1, idx2 = self.pairs[pair_idx]
        K1 = torch.tensor(self.intrinsics[idx1].copy()).float().reshape(3, 3)
        K2 = torch.tensor(self.intrinsics[idx2].copy()).float().reshape(3, 3)

        # read and compute relative poses
        T1 = self.poses[idx1]
        T2 = self.poses[idx2]
        T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1))).float()[:4, :4]

        # Load positive pair data
        im_A, im_B = self.image_paths[idx1], self.image_paths[idx2]
        depth1, depth2 = self.depth_paths[idx1], self.depth_paths[idx2]
        im_A_ref = os.path.join(self.data_root, im_A)
        im_B_ref = os.path.join(self.data_root, im_B)
        depth_A_ref = os.path.join(self.data_root, depth1)
        depth_B_ref = os.path.join(self.data_root, depth2)

        try:
            im_A = self.load_im(im_A_ref)
            im_B = self.load_im(im_B_ref)
            depth_A = self.load_depth(depth_A_ref)
            depth_B = self.load_depth(depth_B_ref)
        except Exception as e:
            print("Error loading image or depth:", e)
            return None

        K1 = self.scale_intrinsic(K1, im_A.width, im_A.height)
        K2 = self.scale_intrinsic(K2, im_B.width, im_B.height)

        # Process images
        try:
            if self.use_randaug:
                im_A, im_B = self.rand_augment(im_A, im_B)

            im_A, im_B = self.im_transform_ops((im_A, im_B))
            depth_A, depth_B = self.depth_transform_ops(
                (depth_A[None, None], depth_B[None, None])
            )

            [im_A, im_B, depth_A, depth_B], t = self.rand_shake(
                im_A, im_B, depth_A, depth_B)
            K1[:2, 2] += t
            K2[:2, 2] += t

            im_A, im_B = im_A[None], im_B[None]
            if self.random_eraser is not None:
                im_A, depth_A = self.random_eraser(im_A, depth_A)
                im_B, depth_B = self.random_eraser(im_B, depth_B)

            if self.use_horizontal_flip_aug:
                if np.random.rand() > 0.5:
                    im_A, im_B, depth_A, depth_B, K1, K2 = self.horizontal_flip(
                        im_A, im_B, depth_A, depth_B, K1, K2)

            if self.use_single_horizontal_flip_aug:
                if np.random.rand() > 0.5:
                    im_B, depth_B, K2 = self.single_horizontal_flip(
                        im_B, depth_B, K2)
        except Exception as e:
            print(
                f"Error in transform ({self.image_paths[idx1]}, {self.image_paths[idx1]}):", e)
            return None

        data_dict = {
            "im_A": im_A[0],
            "im_A_identifier": self.image_paths[idx1].split("/")[-1].split(".jpg")[0],
            "im_B": im_B[0],
            "im_B_identifier": self.image_paths[idx2].split("/")[-1].split(".jpg")[0],
            "im_A_depth": depth_A[0, 0],
            "im_B_depth": depth_B[0, 0],
            "K1": K1,
            "K2": K2,
            "T_1to2": T_1to2,
            "im_A_path": im_A_ref,
            "im_B_path": im_B_ref,
        }

        return data_dict


class ExtreBuilder:
    def __init__(self, data_root: str = "./data/extre") -> None:
        self.data_root = data_root
        self.scene_info_root = os.path.join(data_root, "scene_info")
        self.all_scenes = set(os.listdir(self.scene_info_root))
        self.test_scenes = {"Madrid4_117@-83@276@68@0@90.npz",
                            "Madrid4_90@-33@76@58@0@90.npz",
                            "Madrid1_93@467@65@51@0@90.npz",
                            "Berlin6_141@17@21@70@0@90.npz",
                            "Tokyo5_92@167@326@64@0@90.npz",
                            "Madrid1_93@-233@-385@59@0@90.npz",
                            "German5_61@-263@139@63@0@90.npz",
                            "Milano3_123@-434@318@58@0@90.npz",
                            "NewYork4_138@-133@169@66@0@90.npz",
                            "Bern0_143@216@-387@51@0@90.npz",
                            "Berlin0_111@-133@-280@52@0@90.npz",
                            "Madrid0_122@167@215@51@0@90.npz",
                            "Milano2_134@116@218@51@0@90.npz"}
        self.ignore_scenes = set()

    def build_scenes(self, split: str = "train", **kwargs):
        if split == "train":
            scene_names = self.all_scenes - self.test_scenes - self.ignore_scenes
        elif split == "test":
            scene_names = self.test_scenes - self.ignore_scenes
        else:
            raise ValueError(f"Unknown split {split}")

        scenes = []
        for scene_name in scene_names:
            if ".npz" not in scene_name:
                continue
            scene_info_path = os.path.join(self.scene_info_root, scene_name)
            scene_info = np.load(scene_info_path, allow_pickle=True)
            scene = ExtreScene(
                data_root=self.data_root,
                scene_info=scene_info,
                scene_name=scene_name,
                **kwargs
            )
            if len(scene) != 0:
                scenes.append(scene)
        return scenes

    def weight_scenes(self, concat_dataset, alpha: float = 0.5) -> torch.Tensor:
        ns = []
        for d in concat_dataset.datasets:
            ns.append(len(d))
        ws = torch.cat([torch.ones(n) / n**alpha for n in ns])
        return ws
