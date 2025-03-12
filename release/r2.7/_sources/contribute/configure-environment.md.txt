# Configure a development environment

The goal of this guide is to set up an interactive development
environment on a Cloud TPU with PyTorch/XLA installed. If this is your
first time using TPUs, we recommend you start with
[Colab](https://colab.sandbox.google.com/github/tensorflow/docs/blob/master/site/en/guide/tpu.ipynb)
and [Kaggle](https://www.kaggle.com/discussions/product-feedback/369338)
or. Both options have PyTorch/XLA preinstalled with dependencies and
ecosystem packages. For an up-to-date list of examples, see our main
[README](https://github.com/pytorch/xla).

If you would like to set up a more customized development environment,
keep reading.

## Visual Studio Code

Prerequisites:

-   [Visual Studio Code](https://code.visualstudio.com/download) with
    the [Remote Development
    extensions](https://code.visualstudio.com/docs/remote/remote-overview)
    installed on your local machine
-   A GCP project with Cloud TPU quota. For more information about
    requesting Cloud TPU quota, see the [official
    documentation](https://cloud.google.com/tpu/docs/quota)
-   An SSH key registered with `ssh-agent`. If you have not already done
    this, see [GitHub's
    documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

Before you begin, export environment variables with the GCP project and
zone where you have Cloud TPU quota:

``` bash
export PROJECT=...
export ZONE=...
export TPU_TYPE=... # e.g. "v2-8"
```

### Creating and connecting to your TPU

Create a Cloud TPU VM with your SSH key registered:

``` bash
# Assuming your SSH key is named `id_ed25519`
gcloud compute tpus tpu-vm create --project=$PROJECT --zone=$ZONE --accelerator-type=$TPU_TYPE --version=tpu-ubuntu2204-base --metadata="ssh-keys=$USER:$(cat ~/.ssh/id_ed25519.pub)" $USER-tpu
```

Check that your TPU has an external IP and SSH to it:

``` bash
gcloud compute tpus tpu-vm describe --project=$PROJECT --zone=$ZONE $USER-tpu --format="value(networkEndpoints.accessConfig.externalIp)"
# Output: 123.123.123.123
```

Give your TPU a friendly name to make future steps easier:

``` bash
echo -e Host $USER-tpu "\n " HostName $(gcloud compute tpus tpu-vm describe --project=$PROJECT --zone=$ZONE $USER-tpu --format="value(networkEndpoints.accessConfig.externalIp)") >> ~/.ssh/config
```

SSH to your TPU to test your connection:

``` bash
ssh $USER-tpu
```

### Setting up a Visual Studio Code workspace with PyTorch/XLA

From the [VS Code Command
Palette](https://code.visualstudio.com/docs/getstarted/userinterface#_command-palette),
select `` `Remote-SSH: Connect to Host ``
\<<https://code.visualstudio.com/docs/remote/ssh>\>[\_\_ and select the
host you just created (named ]{.title-ref}[\$USER-tpu]{.title-ref}\`).
VS Code will then open a new window connected to your TPU VM.

From the built-in `Terminal`, create a new folder to use as a workspace
(e.g. `mkdir ptxla`). Then open the folder from the UI or Command
Palette.

Note: It is optional (but recommended) at this point to install the
official [Python
extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
and create a [venv virtual
environment](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command)
via the Command Palette (`Python: Create Environment`).

Install the latest PyTorch and PyTorch/XLA releases:

``` bash
pip install numpy torch torch_xla[tpu] \
  -f https://storage.googleapis.com/libtpu-wheels/index.html \
  -f https://storage.googleapis.com/libtpu-releases/index.html
```

Create a file `test.py`:

``` python
import torch_xla as xla

# Optional
xla.runtime.set_device_type("TPU")

print("XLA devices:", xla.real_devices()) 
```

Run the test script from your terminal:

``` bash
$ python test.py
# Output: XLA devices: ['TPU:0', 'TPU:1', 'TPU:2', 'TPU:3', 'TPU:4', 'TPU:5', 'TPU:6', 'TPU:7']
# Number of devices will vary based on TPU type
```

### Next steps

That's it! You should now have a remote Visual Studio Code workspace set
up with PyTorch/XLA installed. To run more realistic examples, see our
[examples guide](https://github.com/pytorch/xla/tree/master/examples).
