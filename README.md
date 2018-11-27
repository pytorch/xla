# How To Build And Run PyTorch For TPU

To build:

* Build PyTorch from source __with `pytorch.patch` applied__, following the regular [instructions](https://github.com/pytorch/pytorch#from-source).
* Clone this repository in the root folder of the PyTorch sources used for the previous step.
  Run `git submodule update --init` to get the third-party dependencies and `python setup.py install` to build and install the extension.

To run the tests, follow __one__ of the options below:

* Run on CPU using the local client:

  `export XLA_USE_XRT=0 export XLA_GRPC_HOST="" XLA_PLATFORM="CPU"`

* Run on CPU using the XRT client:

  `export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localhost/replica:0/task:0/device:XLA_CPU:0" XRT_WORKERS="localhost:0;"`

* Run on TPU using the XRT client:

  `export XLA_USE_XRT=1 XRT_DEVICE_MAP="TPU:0;/job:tpu_worker/replica:0/task:0/device:TPU:0" XRT_WORKERS="tpu_worker:0;grpc://localhost:51000"`. Specify the TPU node by doing __one__ of the following:

  - create a `$HOME/.pytorch_tpu.conf` file with the following content: `worker: tpu_worker <ip of the tpu node>:8470`

  - set the `XRT_TPU_CONFIG` environment variable: `export XRT_TPU_CONFIG="tpu_worker;0;<ip of the tpu node>:8470"`.



Then run `python test/test_operations.py`. Some of the tests are currently skipped.
