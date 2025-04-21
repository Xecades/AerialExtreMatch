import torch
import tqdm
import romatch
from romatch.datasets.mixed import get_mixed_dataset
import romatch.utils.writer as writ
import pytorch_lightning as pl
from romatch.utils.plotting import visualize_matches_roma
from romatch.utils.collate import collate_fn_replace_corrupted
from functools import partial


class MixedVisualizeBenchmark:
    def __init__(self, h=384, w=512) -> None:
        pl.seed_everything(2333)

        self.dataset, self.ws = get_mixed_dataset(
            h, w, train=False, mega_percent=0.1)
        self.batch_size = 8

        collate_fn = partial(
            collate_fn_replace_corrupted,
            dataset=self.dataset
        )
        self.sampler = torch.utils.data.WeightedRandomSampler(
            self.ws, replacement=False, num_samples=100
        )
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.batch_size,
            sampler=self.sampler,
            collate_fn=collate_fn,
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

            matches_f, certainty_f = fine
            matches_c, certainty_c = coarse

            fig_fs = []
            fig_cs = []
            for b in tqdm.trange(self.batch_size):
                mf, _ = model.sample(matches_f[b], certainty_f[b], num=10000)
                mc, _ = model.sample(matches_c[b], certainty_c[b], num=10000)

                fig_f = visualize_matches_roma(
                    im_A=im_A[b].cpu(),
                    im_B=im_B[b].cpu(),
                    warp=mf.cpu(),
                    T_1to2=T_1to2[b].cpu(),
                    K1=K1[b].cpu(),
                    K2=K2[b].cpu(),
                    threshold=1e-4,
                    num_points=300,
                    dpi=300
                )
                fig_c = visualize_matches_roma(
                    im_A=im_A[b].cpu(),
                    im_B=im_B[b].cpu(),
                    warp=mc.cpu(),
                    T_1to2=T_1to2[b].cpu(),
                    K1=K1[b].cpu(),
                    K2=K2[b].cpu(),
                    threshold=1e-4,
                    num_points=300,
                    dpi=300
                )
                fig_fs.append(fig_f)
                fig_cs.append(fig_c)

            if not romatch.TEST_MODE:
                writ.writer.add_figure(
                    tag=f"visualization_fine",
                    figure=fig_fs,
                    global_step=romatch.GLOBAL_STEP
                )
                writ.writer.add_figure(
                    tag=f"visualization_coarse",
                    figure=fig_cs,
                    global_step=romatch.GLOBAL_STEP
                )
                writ.writer.flush()
            else:
                for i in range(self.batch_size):
                    fig_fs[i].savefig(
                        f"vis/visualize_fine_{romatch.GLOBAL_STEP}_{i}.png")
                    fig_cs[i].savefig(
                        f"vis/visualize_coarse_{romatch.GLOBAL_STEP}_{i}.png")
