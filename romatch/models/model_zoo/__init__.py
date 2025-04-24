from typing import Union
from .roma_models import roma_model
import torch

weight_urls = {
    "romatch": {
        # TODO: add the URL for the external model
        "extre": "file:///home/local/Develop/ExtreRoMa/workspace/models/roma_extre.714912.pth",
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}


def roma_extre(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16
):
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    if str(device) == "cpu":
        amp_dtype = torch.float32

    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["extre"],
            map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"],
            map_location=device
        )

    model = roma_model(
        resolution=coarse_res,
        upsample_preds=True,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype
    )

    model.upsample_res = upsample_res
    print(
        f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model
