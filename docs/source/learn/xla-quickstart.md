# Quickstart: Your First PyTorch/XLA Model

This guide will walk you through training a basic PyTorch model on an XLA device. We'll use the classic MNIST dataset and a simple convolutional neural network (CNN). By the end of this quickstart, you'll see how few modifications are needed to get your PyTorch code running with PyTorch/XLA.

## Prerequisites

Before you start, please ensure you have:

1. Successfully completed the [Installation Guide](TODO) and have PyTorch/XLA installed and configured for your target XLA device (e.g., TPU or GPU).
2. Basic familiarity with PyTorch concepts (tensors, `nn.Module`, `DataLoader`, optimizers).

## The MNIST Training Script

Install `torchvision` to load the built-in MNIST dataset, and create a `data` directory to store it.

```shell
pip install torchvision
mkdir data
```

Create a Python script named `mnist_xla_quickstart.py`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# PyTorch/XLA specific imports
import torch_xla
import torch_xla.core.xla_model as xm

# Define the CNN Model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7*7*64, 128) # Adjusted for 28x28 image, 2 pooling layers
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7*7*64) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_mnist():
    # Training parameters
    epochs = 1
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 64

    # 1. Acquire the XLA device
    device = xm.xla_device()
    print(f"Running on XLA device: {device}")

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    # 2. Initialize the model and move it to the XLA device
    model = MNISTNet().to(device)

    # Define loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            # 3. Move data and target to the XLA device
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()

            optimizer.step()

            # 4. Synchronize: Tell XLA to execute the accumulated operations
            # For single device training, torch_xla.sync() is often used.
            # For multi-device training (covered later), xm.optimizer_step(optimizer)
            # also performs this synchronization.
            torch_xla.sync()

            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    print("Training finished!")

if __name__ == '__main__':
    train_mnist()
```

## Running the Script

1. Save the code above as `mnist_xla_quickstart.py`.
2. Ensure your environment is configured to use your XLA device (e.g., `PJRT_DEVICE=TPU` or `PJRT_DEVICE=CUDA` set as environment variables if not already configured globally).
3. Run the script from your terminal:

```shell
python mnist_xla_quickstart.py
```

You should see output indicating the XLA device (index) being used and training progress, including loss values.

```shell
Running on XLA device: xla:0
100.0%
100.0%
100.0%
100.0%
Starting training...
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.303487
Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.702035
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.492530
Train Epoch: 1 [19200/60000 (32%)]	Loss: 0.294703
Train Epoch: 1 [25600/60000 (43%)]	Loss: 0.191667
Train Epoch: 1 [32000/60000 (53%)]	Loss: 0.233557
Train Epoch: 1 [38400/60000 (64%)]	Loss: 0.135758
Train Epoch: 1 [44800/60000 (75%)]	Loss: 0.257190
Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.121358
Train Epoch: 1 [57600/60000 (96%)]	Loss: 0.073349
Training finished!
```

## Explanation of XLA-Specific Parts

Let's break down the PyTorch/XLA specific lines:

1. **`import torch_xla`** and **`import torch_xla.core.xla_model as xm`**: These lines import the necessary PyTorch/XLA modules. The `torch_xla` import initializes the XLA backend. `xm` is a common alias for `torch_xla.core.xla_model`, which provides core XLA functionalities.

2. **`device = xm.xla_device()`**: This is the key function to obtain an XLA device object. PyTorch/XLA will automatically select an available XLA device (like a TPU core or a GPU managed by XLA). Tensors and models need to be moved to this device to be accelerated.

3. **`.to(device)`**: Just like in standard PyTorch, you use `.to(device)` to move your model's parameters (`model.to(device)`) and your input data and targets (`data.to(device)`, `target.to(device)`) to the XLA device.

4. **`torch_xla.sync()`**: This is a crucial function in PyTorch/XLA when not using `xm.optimizer_step()` (which is common in multi-device setups). PyTorch/XLA operations are *lazy*; they build up a computation graph behind the scenes. `torch_xla.sync()` tells PyTorch/XLA that the current phase of computation definition is complete. This triggers the XLA compiler to optimize and execute the accumulated graph on the accelerator. It's typically called once per training iteration, often after `optimizer.step()`. In multi-processing scenarios, `xm.optimizer_step(optimizer)` often replaces the separate `optimizer.step()` and `torch_xla.sync()` calls, as it handles gradient synchronization and the step execution.

## Key Takeaways

* **Minimal Code Changes**: Running PyTorch on XLA devices often requires only a few lines of code to be added or modified.
* **Device Agnostic Model Code**: Your core model definition (`MNISTNet`), loss function, and optimizer logic remain standard PyTorch code.
* **Lazy Execution**: PyTorch/XLA defers computation until explicitly synchronized. This allows for powerful graph-level optimizations by the XLA compiler.

## Next Steps

Congratulations! You've run your first PyTorch model on an XLA device.

* To learn how to scale this to multiple XLA devices, explore our guides on [Distributed Training](TODO).
* If you're coming from a GPU background, check out our [Migrating from GPUs to TPUs](./gpu-to-tpu-migration.html) guide for more detailed advice.
* To understand the "why" behind PyTorch/XLA's behavior, dive into [Learn Core Concepts](TODO).
