import argparse
import torch
import torch_xla.utils.checkpoint
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.experimental.xla_sharding as xs
from torch_xla.experimental.xla_sharding import Mesh

def run(grad_checkpoint):
    device = xm.xla_device()
    model = torch.nn.ModuleList(
        [
            torch.nn.Sequential(
                torch.nn.Conv2d(1024, 1024, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(1024, 1024, 1),
                torch.nn.ReLU(),
            )
            for _ in range(64)
        ]
    ).to(device)

    mesh = Mesh([0, 1, 2, 3], (2, 2, 1, 1))
    for n_l, layer in enumerate(model):
        for n_l, seq_layer in enumerate(layer):
            if "Conv" in seq_layer.__str__():
                xs.mark_sharding(seq_layer.weight, mesh, (0,1,2,3))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    for step in range(3):
        dummy_data = torch.zeros(64, 1024, 14, 14, device=device)
        optimizer.zero_grad()
        x = dummy_data
        for n_l, layer in enumerate(model):
            if n_l > 0 and grad_checkpoint:
                x = torch_xla.utils.checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        dummy_loss = x.sum()
        dummy_loss.backward()
        optimizer.step()
        xm.mark_step()
        #print(f"step {step}, free memory = {xm.get_memory_info(device)['kb_free']}")

    print(met.metrics_report())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_checkpoint", type=int, required=True)
    args = parser.parse_args()
    run(args.grad_checkpoint)
