# ExtreData: Lorem Ipsum Dolor Sit Amet
### [Project Page]() | [Paper]()

<br />

> ExtreData: Lorem Ipsum Dolor Sit Amet  
> [John Doe](), [Jane Doe]()  
> NeurIPS 2025

## Introduction

This repo contains the code for training from scratch and finetuning the [RoMa](https://arxiv.org/abs/2305.15404) model on our proposed ExtreData dataset. The code is built on top of the [RoMa repo](https://github.com/Parskatt/RoMa). It is recommended to check out the original repo for more details on the model and its usage. We will only cover the essential changes here.

## Checkpoints

We provide our pretained weights for the ExtreData dataset at [[Link]]().

## Setup / Settings / Usage

Follow the instructions on [RoMa](https://github.com/Parskatt/RoMa).

## Demo

A matching demo is provided in the [demos folder](demo).

See [RoMa](https://github.com/Parskatt/RoMa?tab=readme-ov-file#demo--how-to-use) for more details.

## Reproducing Results

The experiments are implemented in the [experiments folder](experiments).

### Training

1. Follow the instructions on [DKM](https://github.com/Parskatt/DKM/blob/main/docs/training.md#megadepth) for downloading and preprocessing the MegaDepth dataset.
2. Download the [ExtreData]() dataset and unzip it into the `data/extredata` folder.  
   By now you should have the following folder structure in `data`:
    ```
    data
    ├── extredata
    │   ├── scene_info
    │   └── Basaier0, ...
    └── megadepth
        ├── phoenix
        │   └── S6
        │       └── zl548
        │           └── MegaDepth_v1
        ├── prep_scene_info
        └── Undistorted_SfM
    ```
3. Run the relevant experiment. e.g.
    ```bash
    # Finetuning from pretained RoMa weights
    torchrun --nproc_per_node=8 --nnodes=1 --rdzv_backend=c10d experiments/train_roma.py --gpu_batch_size 3 --use_pretained_roma

    # Training from scratch
    torchrun --nproc_per_node=8 --nnodes=1 --rdzv_backend=c10d experiments/train_roma.py --gpu_batch_size 3
    ```

### Testing

TBD.

## License

TBD.

## Acknowledgement

Our codebase builds on the code in [RoMa](https://github.com/Parskatt/RoMa).

## BibTeX

```bibtex
```
