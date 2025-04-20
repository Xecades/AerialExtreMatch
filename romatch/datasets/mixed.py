from romatch.datasets.extredata import ExtredataBuilder
from romatch.datasets.megadepth import MegadepthBuilder
from torch.utils.data import ConcatDataset
import torch


def get_mixed_dataset(h, w, train=True, mega_percent=0.1):
    extredata_scenes, extredata_weight = get_extredata_dataset(h, w, train)
    megadepth_scenes, megadepth_weight = get_megadepth_dataset(h, w, train)

    scenes = ConcatDataset([extredata_scenes, megadepth_scenes])

    extredata_weight = extredata_weight / extredata_weight.sum()
    megadepth_weight = megadepth_weight / megadepth_weight.sum()
    weight = torch.cat(
        [
            extredata_weight * (1 - mega_percent),
            megadepth_weight * mega_percent,
        ]
    )
    # weight.sum() should be 1.0
    return scenes, weight


def get_extredata_dataset(h, w, train=True):
    builder = ExtredataBuilder(data_root="data/extredata")

    if train:
        scense1 = builder.build_scenes(
            split="train",
            min_overlap=0.01,
            ht=h,
            wt=w,
        )
        scenes2 = builder.build_scenes(
            split="train",
            min_overlap=0.35,
            ht=h,
            wt=w,
        )
        scenes = scense1 + scenes2
    else:
        scenes = builder.build_scenes(
            split="test",
            ht=h,
            wt=w,
        )

    scenes = ConcatDataset(scenes)
    weight = builder.weight_scenes(scenes, alpha=0.75)
    return scenes, weight


def get_megadepth_dataset(h, w, train=True):
    builder = MegadepthBuilder(
        data_root="data/megadepth",
        loftr_ignore=True,
        imc21_ignore=True
    )

    if train:
        use_horizontal_flip_aug = True
        rot_prob = 0
        scenes1 = builder.build_scenes(
            split="train_loftr",
            min_overlap=0.01,
            shake_t=32,
            use_horizontal_flip_aug=use_horizontal_flip_aug,
            rot_prob=rot_prob,
            ht=h,
            wt=w,
        )
        scenes2 = builder.build_scenes(
            split="train_loftr",
            min_overlap=0.35,
            shake_t=32,
            use_horizontal_flip_aug=use_horizontal_flip_aug,
            rot_prob=rot_prob,
            ht=h,
            wt=w,
        )
        scenes = scenes1 + scenes2
    else:
        scenes = builder.build_scenes(
            split="test_loftr",
            ht=h,
            wt=w,
        )

    scenes = ConcatDataset(scenes)
    weight = builder.weight_scenes(scenes, alpha=0.75)
    return scenes, weight
