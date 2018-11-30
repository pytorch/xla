# How To Build And Run PyTorch For TPU

To build:

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Checkout the following commit ID:

  ```
  git checkout 1ca0ec7299b9352123a07525320a04e39d657781
  ```

* Clone the _PyTorch/XLA_ repo:

  ```
  git clone --recursive https://github.com/pytorch/xla.git
  ```

* Apply the `pytorch.patch` to the current `xla` code. From within the _pytorch_ source folder:

  ```
  patch -p1 < xla/pytorch.patch
  ```

* Currently _PyTorch_ does not build with GCC 8.x. A known working GCC version is 7.3.x, so install that in your VM:

  ```
  apt-get install gcc-7 g++-7
  export CC=gcc-7
  export CXX=g++-7
  ```
  
* Build _PyTorch_ from source following the regular [instructions](https://github.com/pytorch/pytorch#from-source).

  ```
  python setup.py install
  ```

* Build the _PyTorch/XLA_ source:

  ```
  cd xla/
  python setup.py install
  ```

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
