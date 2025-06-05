import torch
import torch_xla
import os

os.environ['TPU_SKIP_MDS_QUERY'] = 'True'
os.environ['TPU_ACCELERATOR_TYPE'] = 'tpu7x-8'
os.environ['TPU_HOST_BOUNDS'] = '2,2,1'
os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '1,1,1'
os.environ['TPU_WORKER_ID'] = '0'


def run(rank):
    device = torch_xla.core.xla_model.xla_device()
    print(f"Running on device: {device}")

    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    t = []
    for i in range(100):
        z = x + y + i
        t.append(z)

    print(f"Completed computation on rank {rank}")
if __name__ == "__main__":
    torch_xla.launch(run)