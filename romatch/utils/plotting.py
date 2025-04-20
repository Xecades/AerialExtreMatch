import matplotlib.lines
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import bisect
from romatch.utils.utils import imagenet_mean, imagenet_std
from romatch.utils.utils import warp_to_pixel_coords
from kornia.geometry.epipolar import numeric


def grid_sample_points(xs, ys, grid_size=20):
    selected = []

    visited = set()
    for x, y in zip(xs, ys):
        gx = x.item() // grid_size
        gy = y.item() // grid_size
        key = (gx, gy)
        if key not in visited:
            selected.append((x, y))
            visited.add(key)
    return zip(*selected) if selected else ([], [])


def unnormalize(im, mean=imagenet_mean, std=imagenet_std):
    mean = mean[:, None, None].to(im.device)
    std = std[:, None, None].to(im.device)
    return im * std + mean


def denormalize_grid(grid, H, W):
    x = (grid[..., 0] + 1) * 0.5 * W
    y = (grid[..., 1] + 1) * 0.5 * H
    return torch.stack([x, y], axis=-1)  # [..., 2]


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)


def make_matching_figure(
    img0, img1, mkpts0, mkpts1, color,
    kpts0=None, kpts1=None, text=[], dpi=200, path=None
):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                             (fkpts0[i, 1], fkpts1[i, 1]),
                                             transform=fig.transFigure, c=color[i], linewidth=1)
                     for i in range(len(mkpts0))]

        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = "k" if img0[:100, :200].mean() > 200 else "w"
    fig.text(
        0.01, 0.99, "\n".join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va="top", ha="left", color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        return fig


def visualize_matches(data, threshold=1e-4, num_points=1000):
    im0 = unnormalize(data["image0"]).cpu().numpy()
    im1 = unnormalize(data["image1"]).cpu().numpy()
    im0 = (im0.transpose(1, 2, 0) * 255).round().astype(np.int32)
    im1 = (im1.transpose(1, 2, 0) * 255).round().astype(np.int32)
    assert im0.shape == im1.shape

    kpts0 = data["mkpts0_f"].cpu().numpy()
    kpts1 = data["mkpts1_f"].cpu().numpy()
    epi_errs = data["epi_errs"].cpu().numpy()
    correct_mask = epi_errs < threshold

    if len(correct_mask) > num_points:
        indices = np.random.choice(
            len(correct_mask), num_points, replace=False)
        kpts0 = kpts0[indices]
        kpts1 = kpts1[indices]
        epi_errs = epi_errs[indices]
        correct_mask = correct_mask[indices]

    alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, threshold, alpha=alpha)

    return make_matching_figure(
        img0=im0,
        img1=im1,
        mkpts0=kpts0,
        mkpts1=kpts1,
        color=color,
    )


def visualize_matches_roma(im_A, im_B, warp, T_1to2, K1, K2, threshold=1e-4, num_points=1000):
    _, H, W = im_A.shape
    px_warp = warp_to_pixel_coords(warp, H, W, H, W).reshape(-1, 4)
    data = {
        "image0": im_A,
        "image1": im_B,
        "mkpts0_f": px_warp[:, :2],
        "mkpts1_f": px_warp[:, 2:],
        "T_0to1": T_1to2,
        "K0": K1,
        "K1": K2,
    }

    compute_symmetrical_epipolar_errors(data)
    return visualize_matches(
        data=data,
        threshold=threshold,
        num_points=num_points
    )


def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    from kornia.geometry.conversions import convert_points_to_homogeneous

    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) +
                    1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(data):
    bT_0to1 = data["T_0to1"].unsqueeze(0)
    Tx = numeric.cross_product_matrix(bT_0to1[:, :3, 3])
    E_mat = Tx @ bT_0to1[:, :3, :3]

    pts0 = data["mkpts0_f"]
    pts1 = data["mkpts1_f"]

    epi_errs = symmetric_epipolar_distance(
        pts0, pts1, E_mat[0], data["K0"], data["K1"])
    data.update({"epi_errs": epi_errs})


if __name__ == "__main__":
    h, w = (14*8*5, 14*8*5)
    B = 8

    data = torch.load("vis/test.pth")
    mdata = torch.load("vis/match.pth")
    im_A, im_B, depth1, depth2, T_1to2, K1, K2 = (
        data["im_A"].cuda(),
        data["im_B"].cuda(),
        data["im_A_depth"].cuda(),
        data["im_B_depth"].cuda(),
        data["T_1to2"].cuda(),
        data["K1"].cuda(),
        data["K2"].cuda(),
    )
    warp, certainty = mdata["matches"].cuda(), mdata["certainty"].cuda()

    fig = visualize_matches_roma(
        im_A=im_A[1],
        im_B=im_B[1],
        warp=warp[1],
        T_1to2=T_1to2[1],
        K1=K1[1],
        K2=K2[1],
    )
    fig.savefig(f"vis/test.png", dpi=300)
