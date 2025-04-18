from torch.utils.tensorboard import SummaryWriter
import os


class Dummy:
    def __init__(*args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, *args, **kwargs): return self


writer = Dummy()


def init_writer(experiment_name: str, rank: int):
    global writer
    if rank == 0:
        log_dir = os.path.join("workspace/logs/", experiment_name)
        writer = SummaryWriter(log_dir=log_dir)
