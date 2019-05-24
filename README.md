# How To Build And Run PyTorch For TPU

To build:

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Clone the _PyTorch/XLA_ repo:

  ```Shell
  git clone --recursive https://github.com/pytorch/xla.git
  ```

## Building with script:

* To build and install `torch` and `torch_xla`:

  ```Shell
  xla/scripts/build_torch_wheels.sh
  ```

## Building manually:

* If a file named xla/.torch_commit_id exists, use its content to checkout the PyTorch commit ID:

  ```Shell
  git checkout $(cat xla/.torch_commit_id)
  ```

* Apply PyTorch patches:

  ```Shell
  xla/scripts/apply_patches.sh
  ```

* Install the Lark parser used for automatic code generation:

  ```Shell
  pip install lark-parser
  ```

* Currently _PyTorch_ does not build with _GCC_ 6.x, 7.x, and 8.x (various kind of ICEs). _CLANG_ 7.x is known to be working, so install that in your VM:

  ```Shell
  sudo apt-get install clang-7 clang++-7
  export CC=clang-7 CXX=clang++-7
  ```

  You may need to add the following line to your _/etc/apt/sources.list_ file:

  ```Shell
  deb http://deb.debian.org/debian/ testing main
  ```

  And run the following command before trying again to install _CLANG_:

  ```Shell
  sudo apt-get update
  ```

* Build _PyTorch_ from source following the regular [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  python setup.py install
  ```

* Install Bazel following the [instructions](https://docs.bazel.build/versions/master/install.html). You should be installing version >= 0.24.1.

* Build the _PyTorch/XLA_ source:

  ```Shell
  cd xla/
  python setup.py install
  ```

To run the tests, follow __one__ of the options below:

* Run on local CPU using the XRT client:

  ```Shell
  export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  ```

  Select any free TCP port you prefer instead of 40934 (totally arbitrary).

* Run on Cloud TPU using the XRT client, use one of the following:

  - Set the XRT_TPU_CONFIG environment variable:

    ```Shell
    export XRT_TPU_CONFIG="tpu_worker;0;<IP of the TPU node>:8470"
    ```

  - Create a `$HOME/.pytorch_tpu.conf` file with the following content: `worker: tpu_worker <IP of the TPU node>:8470`


Note that the IP of the TPU node can change if the TPU node is reset. If _PyTorch_
seem to hang at startup, verify that the IP of your TPU node is still the same of
the one you have configured.

If you are planning to be building from source and hence using the latest _PyTorch/TPU_ code base,
it is suggested for you to select the _Nightly_ builds when you create a Cloud TPU instance.

Then run `test/run_tests.sh` and `test/cpp/run_tests.sh` to verify the setup is working.


[![CircleCI](https://circleci.com/gh/pytorch/xla.svg?style=svg)](https://circleci.com/gh/pytorch/xla)



# How To Install Pre Built PyTorch TPU Wheels

It is recommended to use Conda environments to isolate _PyTorch/TPU_ packages from the others.
To install Anaconda follow the [instructions](https://docs.anaconda.com/anaconda/install/).
Then create an environment dedicated to _PyTorch/TPU_ and activate it (activation should happen every time you want to work in such environment):

```Shell
conda create --name pytorch_tpu --clone base
source activate pytorch_tpu
```

Install the _gsutil_ package to allow access to _GCS_ (Google Cloud Storage) following the [instructions](https://cloud.google.com/storage/docs/gsutil_install).

Then run:

```Shell
scripts/update_torch_wheels.sh
```

The same script can be run again when you want to update the _PyTorch/TPU_ wheels.
