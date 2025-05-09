import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L

encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))


class LitAutoEncoder(L.LightningModule):

  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder, self.decoder = encoder, decoder

  def training_step(self, batch, batch_idx):
    x, y = batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = nn.functional.mse_loss(x_hat, x)
    self.log("train_loss", loss)
    return loss

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=1e-3)


dataset = tv.datasets.MNIST(
    ".", download=True, transform=tv.transforms.ToTensor())

# Lightning will automatically use all available GPUs!
trainer = L.Trainer()
# trainer.fit(LitAutoEncoder(encoder, decoder), data.DataLoader(dataset, batch_size=64))

# ==== above is the lightning example from
# https://lightning.ai/pytorch-lightning

import torchax
from torchax.interop import jax_view, torch_view
import jax
import optax


class JaxTrainer:

  def __init__(self):
    pass

  def torch_opt_to_jax_opt(self, torch_opt):
    # TODO: Can convert optimizer instead of using a jax one
    return optax.adam(0.001)

  def fit(self, lightning_mod, data_loader):

    xla_env = torchax.default_env()

    def lightning_mod_loss(weights: jax.Array, data: jax.Array, batch_id):
      """returns loss"""
      weights, data = torch_view((weights, data))
      lightning_mod.load_state_dict(weights, assign=True)
      with xla_env:
        loss = lightning_mod.training_step(data, batch_id)
      return jax_view(loss)

    jax_weights = jax_view(xla_env.to_xla(lightning_mod.state_dict()))
    jax_optimizer = self.torch_opt_to_jax_opt(
        lightning_mod.configure_optimizers())
    opt_state = jax_optimizer.init(jax_weights)
    grad_fn = jax.jit(jax.value_and_grad(lightning_mod_loss))

    for bid in range(3):
      for item in data_loader:
        xla_data = jax_view(xla_env.to_xla(item))
        loss, grads = grad_fn(jax_weights, xla_data, bid)
        updates, opt_state = jax_optimizer.update(grads, opt_state)
        jax_weights = optax.apply_updates(jax_weights, updates)
        print('current_loss', loss)


print('-----------------')
trainer_jax = JaxTrainer()
trainer_jax.fit(
    LitAutoEncoder(encoder, decoder), data.DataLoader(dataset, batch_size=64))
