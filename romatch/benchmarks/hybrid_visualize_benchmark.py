import torch
import tqdm
import romatch
from romatch.datasets import MegadepthBuilder
import romatch.utils.writer as writ
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset
from romatch.utils.plotting import visualize_matches_roma
from romatch.utils.collate import collate_fn_replace_corrupted
from functools import partial


class HybridVisualizeBenchmark:
    def __init__(
        self,
        data_root="data/megadepth",
        h=384,
        w=512,
        seed=2333
    ) -> None:
        pl.seed_everything(seed)

        mega = MegadepthBuilder(data_root=data_root)
        self.dataset = ConcatDataset(
            mega.build_scenes(split="test_loftr", ht=h, wt=w)
        )
        self.batch_size = 4

        collate_fn = partial(
            collate_fn_replace_corrupted,
            dataset=self.dataset
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
        )
        self.data = next(iter(self.dataloader))

    def benchmark(self, model):
        model.train(False)
        with torch.no_grad():
            im_A, im_B, T_1to2, K1, K2 = (
                self.data["im_A"],
                self.data["im_B"],
                self.data["T_1to2"],
                self.data["K1"],
                self.data["K2"],
            )
            fine, coarse = model.match(
                im_A,
                im_B,
                batched=True,
                coarse_result=True
            )

            matches_f, _ = model.sample(fine[0], fine[1], num=10000)
            matches_c, _ = model.sample(coarse[0], coarse[1], num=10000)

            for b in tqdm.trange(self.batch_size):
                fig_f = visualize_matches_roma(
                    im_A=im_A[b].cpu(),
                    im_B=im_B[b].cpu(),
                    warp=matches_f[b].cpu(),
                    T_1to2=T_1to2[b].cpu(),
                    K1=K1[b].cpu(),
                    K2=K2[b].cpu(),
                    threshold=1e-4,
                    num_points=300,
                    dpi=100
                )
                writ.writer.add_figure(
                    tag=f"visualization_fine/{b}",
                    figure=fig_f,
                    global_step=romatch.GLOBAL_STEP
                )

                fig_c = visualize_matches_roma(
                    im_A=im_A[b].cpu(),
                    im_B=im_B[b].cpu(),
                    warp=matches_c[b].cpu(),
                    T_1to2=T_1to2[b].cpu(),
                    K1=K1[b].cpu(),
                    K2=K2[b].cpu(),
                    threshold=1e-4,
                    num_points=300,
                    dpi=100
                )
                writ.writer.add_figure(
                    tag=f"visualization_coarse/{b}",
                    figure=fig_c,
                    global_step=romatch.GLOBAL_STEP
                )
                writ.writer.flush()
