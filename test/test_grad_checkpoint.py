import torch
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla
import torch_xla.utils.checkpoint as checkpoint
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--test_autocast', action='store_true')
FLAGS, leftovers = parser.parse_known_args()


def run():
  device = xm.xla_device()
  model = torch.nn.ModuleList([
      torch.nn.Sequential(
          torch.nn.Conv2d(1024, 1024, 1),
          torch.nn.ReLU(),
          torch.nn.Conv2d(1024, 1024, 1),
          torch.nn.ReLU(),
      ) for _ in range(2)
  ]).to(device)
  optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

  for step in range(20):
    dummy_data = torch.zeros(64, 1024, 14, 14, device=device)
    optimizer.zero_grad()
    x = dummy_data
    if FLAGS.test_autocast:
      with torch.autocast("xla"):
        for n_l, layer in enumerate(model):
          if n_l > 0:
            x = checkpoint.checkpoint(layer, x)
          else:
            x = layer(x)
    else:
      for n_l, layer in enumerate(model):
        if n_l > 0:
          x = checkpoint.checkpoint(layer, x)
        else:
          x = layer(x)

    dummy_loss = x.sum()
    dummy_loss.backward()
    optimizer.step()
    xm.mark_step()


if __name__ == "__main__":
  run()
