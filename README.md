# How To Build And Run PyTorch For TPU

To build:

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Clone the _PyTorch/XLA_ repo:

  ```
  git clone --recursive https://github.com/pytorch/xla.git
  ```

* If a file named xla/.torch_commit_id exists, use its content to checkout the PyTorch commit ID:

  ```
  git checkout $(cat xla/.torch_commit_id)
  ```

* Apply PyTorch patches:

  ```
  xla/scripts/apply_patches.sh
  ```

* Install the Lark parser used for automatic code generation:

  ```
  pip install lark-parser
  ```

* Currently _PyTorch_ does not build with _GCC_ 6.x, 7.x, and 8.x (various kind of ICEs). _CLANG_ 7.x is known to be working, so install that in your VM:

  ```
  sudo apt-get install clang-7 clang++-7
  export CC=clang-7 CXX=clang++-7
  ```

  You may need to add the following line to your _/etc/apt/sources.list_ file:

  ```
  deb http://deb.debian.org/debian/ testing main
  ```
  
  And run the following command before trying again to install _CLANG_:
  
  ```
  sudo apt-get update
  ```

* Build _PyTorch_ from source following the regular [instructions](https://github.com/pytorch/pytorch#from-source).

  ```
  python setup.py install
  ```

* Install Bazel following the [instructions](https://docs.bazel.build/versions/master/install.html)

* Build the _PyTorch/XLA_ source:

  ```
  cd xla/
  python setup.py install
  ```

To run the tests, follow __one__ of the options below:

* Run on local CPU using the XRT client:

  ```
  export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  ```

  Select any free TCP port you prefer instead of 40934 (totally arbitrary).

* Run on Cloud TPU using the XRT client, use one of the following:

  - Set the XRT_TPU_CONFIG environment variable:

    ```
    export XRT_TPU_CONFIG="tpu_worker;0;<IP of the TPU node>:8470"
    ```

  - Create a `$HOME/.pytorch_tpu.conf` file with the following content: `worker: tpu_worker <IP of the TPU node>:8470`


Note that the IP of the TPU node can change if the TPU node is reset. If _PyTorch_
seem to hang at startup, verify that the IP of your TPU node is still the same of
the one you have configured.


Then run `python test/test_operations.py`. Some of the tests are currently skipped.
