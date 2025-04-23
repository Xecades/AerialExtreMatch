from PIL import Image
import numpy as np
from romatch.utils.utils import tensor_to_pil
from romatch import roma_extre

if True:
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import torch
    import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = torch.device("mps")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--im_A_path", default="demo/assets/berlin_A.jpg", type=str)
    parser.add_argument(
        "--im_B_path", default="demo/assets/berlin_B.jpg", type=str)
    parser.add_argument(
        "--save_path", default="demo/roma_warp_berlin.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_extre(
        device=device,
        coarse_res=560,
        upsample_res=(864, 1152),
    )

    H, W = roma_model.get_output_resolution()
    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)

    x1 = torch.tensor(np.array(im1)).permute(2, 0, 1).to(device) / 255
    x2 = torch.tensor(np.array(im2)).permute(2, 0, 1).to(device) / 255

    # Compute masked image warp
    im2_transfer_rgb = F.grid_sample(
        x2[None], warp[:, :W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]

    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
    white_im = torch.ones((H, 2 * W), device=device)
    masked_im = certainty * warp_im + (1 - certainty) * white_im

    # Add original image for reference
    ori_im = torch.cat((x1, x2), dim=2)
    vis_im = torch.cat((ori_im, masked_im), dim=1)

    tensor_to_pil(vis_im, unnormalize=False).save(save_path)
