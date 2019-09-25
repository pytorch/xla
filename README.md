# How to Run PyTorch with TPUs

First, create your [TPU](https://pantheon.corp.google.com/compute/tpus) node with the corresponding release you wish to consume (TPU software version: `pytorch-0.1`):

Once you've created a Cloud TPU node, you can train your PyTorch models by either:

* [Consuming prebuilt docker images (*recommended*)](#consume-prebuilt-docker-images)
* [Consuming prebuilt Compute VM Images](#consume-prebuilt-compute-vm-images)

## Consume Prebuilt Docker Images

Follow these steps to train a PyTorch model with Docker on a TPU:

1. Create a Compute VM and install docker (or use COS VM image)
    * *Note: make sure the Compute VM is within the **same** zone as the TPU node you created or else performance will suffer, also ideally create a VM that has at least 16 cores (`n1-standard-16`) to not be VM compute/network bound.*

    Docker images with `torch` and `torch_xla` preinstalled in the `pytorch` conda
    environment are distributed under: `gcr.io/tpu-pytorch/xla`.

2. SSH into the VM and pull the stable docker image into the VM:

    ```Shell
    (vm)$ docker pull gcr.io/tpu-pytorch/xla:r0.1
    ```

    Note we do also expose the following nightly Docker image versions, but we recommend you use a stable version (`r0.1`):
    * `gcr.io/tpu-pytorch/xla:nightly`
    * `gcr.io/tpu-pytorch/xla:nightly_YYYYMMDD (e.g.: gcr.io/tpu-pytorch/xla:nightly_20190531)`

    If you decide to consume this, be sure to create a TPU with `pytorch-nightly` version.

3. Where `$TPU_IP_ADDRESS` (e.g.: `10.1.1.2`) is your TPU Internal IP displayed in GCP UI, after pulling the docker image you can either:

    * Run the container with a single command:
      ```Shell
      (vm)$ docker run --shm-size 16G -e XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470" gcr.io/tpu-pytorch/xla:r0.1 python /pytorch/xla/test/test_train_mnist.py
      ```

    * Run the script in an interactive shell:
      ```Shell
      (vm)$ docker run -it --shm-size 16G gcr.io/tpu-pytorch/xla:r0.1
      (pytorch) root@CONTAINERID:/$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
      (pytorch) root@CONTAINERID:/$ python pytorch/xla/test/test_train_mnist.py
      ```

## Consume Prebuilt Compute VM Images

1. Create a Compute VM with PyTorch/XLA Image.

    * In the GCP Console, go to the [**VM Instances**](https://console.cloud.google.com/compute/instances) page.
    * Click **Create Instance**.
    * Make sure the compute VM is within the **same** zone as the TPU node you created or else performance will suffer, also ideally create a VM that has at least 16 cores (`n1-standard-16`) to not be VM compute/network bound.
    * In the **Boot disk** section, click **Change** to choose our PyTorch/XLA image.
    * At the bottom of the **OS Images** tab select the **Debian GNU/Linux 9 Stretch + PyTorch/XLA** image.
    * Chose an appropriate dist size based on your dataset and click **Select**.
    * Click **Create** to create the instance.


2. SSH into VM and activate the conda environment you wish to use. Each release (e.g.: `0.1`, `nightly`) is a separate conda environment.

    ```Shell
    (vm)$ export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    (vm)$ conda env list
    # conda environments:
    #
    base                  *  /anaconda3
    pytorch-0.1              /anaconda3/envs/pytorch-0.1
    pytorch-nightly          /anaconda3/envs/pytorch-nightly

    (vm)$ conda activate pytorch-0.1
    (pytorch-0.1)$ cd /usr/share/torch-xla-0.1/pytorch/xla
    (pytorch-0.1)$ python test/test_train_mnist.py
    ```

    To update the wheels `torch` and `torch_xla` to the latest nightly
    distribution (only updates your pytorch-nightly conda env), run:
    ```Shell
    (vm)$ cd /usr/share/torch-xla-nightly/pytorch/xla
    (vm)$ . ./scripts/update_nightly_torch_wheels.sh
    ```

---

# How to Run on TPU Pods (distributed training)

The recommended setup for running distributed training on TPU Pods uses the
pairing of Compute VM [Instance
Groups](https://cloud.google.com/compute/docs/instance-groups/) and TPU Pods.
Each of the Compute VM in the instance group drives 8 cores on the TPU Pod and
so using an instance group ensures each of the Compute VMs use the identical
base image.

Training on pods can be broken down to largely 3 different steps:
1. [Create your instance group (*recommended*)](#create-your-instance-group) or [Use a list of
   VM instances](#list-of-vms)
2. [Create your TPU Pod](#create-your-tpu-pod)
3. [Start distributed training](#start-distributed-training)

## Create your instance group
1. Create an instance template.
* If you have already have a VM instance running that you used to train PyTorch/TPU workloads and want to use that exact setup for distributed training: [instructions](https://cloud.google.com/compute/docs/instance-templates/create-instance-templates#based-on-existing-instance).
* Or, you can create an instance template using the PyTorch/XLA VM image we provide: [instructions](https://cloud.google.com/compute/docs/instance-templates/create-instance-templates#creating_a_new_instance_template).
2. Create an instance group to drive the TPU pod.
* This instance group is where all the input pipeline happens and where we feed all the tensors into the TPUs for training.
* Use the instance template created in step (1) to create your instance group.
* Make sure to (a) create the instance group in a single zone (same zone as the TPU Pod you'll create), (b) no autoscaling or health-checks, (c) number of instances (size of instance group) should be number of cores / 8 (ex. for a v3-32 you'd create an instance group of size 32/8 = 4).
* Here are the instructions for creating an instance group: [instructions](https://cloud.google.com/compute/docs/instance-groups/creating-groups-of-managed-instances#create_managed_group).

## Create your TPU Pod
1. [Create](https://pantheon.corp.google.com/compute/tpus) a TPU pod (same as creating regular TPUs, just select more cores when selecting TPU type).
* Make sure that the TPU is in the same zone as the instance group.
* Make sure that the size of your instance group follows: # instances in group = number of TPU cores / 8.

## Start distributed training
1. SSH into any of the VMs in the instance group and get in an environment where you have `torch` and `torch_xla` installed (whether that's a [conda environment](#consume-prebuilt-compute-vm-images) or [docker container](#consume-prebuilt-docker-images)).
2. Let's say the command you ran to run a v3-8 was: `XLA_USE_BF16=1 python test/test_train_imagenet.py --fake_data`.
* To distribute training as a conda environment process:
```
(pytorch-nightly)$ cd /usr/share/torch-xla-nightly/pytorch/xla
(pytorch-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --conda-env=pytorch-nightly --env=XLA_USE_BF16=1 -- python test/test_train_imagenet.py --fake_data
```

* Or, to distribute training as a docker container:
```
(pytorch-nightly)$ cd /usr/share/torch-xla-nightly/pytorch/xla
(pytorch-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --docker-image=gcr.io/tpu-pytorch/xla:nightly --docker-run-flag=--rm=true --docker-run-flag=--shm-size=50GB --env=XLA_USE_BF16=1 -- python test/test_train_imagenet.py --fake_data
```

## List of VMs
If you up to not use an [instance group](#create-your-instance-group), you can decide to use a list of VM instances that you may have already created (or can create individually). Make sure that you create all the VM instances in the same zone as the TPU node, and also make sure that the VMs have the same configuration (datasets, VM size, disk size, etc.). Then you can [start distributed training](#start-distributed-training) after creating your TPU pod. The difference is in the `python -m torch_xla.distributed.xla_dist` command. For example, to use a list of VMs run the following command (ex. conda with v3-32):
```
(pytorch-nightly)$ cd /usr/share/torch-xla-nightly/pytorch/xla
(pytorch-nightly)$ python -m torch_xla.distributed.xla_dist --tpu=$TPU_POD_NAME --vm $VM1 --vm $VM2 --vm $VM3 --vm $VM4 --conda-env=pytorch-nightly --env=XLA_USE_BF16=1 -- python test/test_train_imagenet.py --fake_data
```

To learn more about TPU Pods check out this [blog
post](https://cloud.google.com/blog/products/ai-machine-learning/googles-scalable-supercomputers-for-machine-learning-cloud-tpu-pods-are-now-publicly-available-in-beta).

---

# How To Build And Run PyTorch For TPU

To build from source:

* Clone the _PyTorch_ repo as per [instructions](https://github.com/pytorch/pytorch#from-source).

  ```Shell
  git clone --recursive https://github.com/pytorch/pytorch
  cd pytorch/
  ```

* Clone the _PyTorch/XLA_ repo:

  ```Shell
  git clone --recursive https://github.com/pytorch/xla.git
  ```

## Building docker image

* We provide a Dockerfile in `docker/` that you can use to build images as the
  following:

  ```Shell
  docker build -t torch-xla -f docker/Dockerfile .
  ```

## Building with script

* To build and install `torch` and `torch_xla`:

  ```Shell
  xla/scripts/build_torch_wheels.sh
  ```

## Building manually

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

* Install Bazel following the [instructions](https://docs.bazel.build/versions/master/install.html). You should only install version 0.24.1, as no older nor newer releases will be able to build the required dependencies.

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

* Run on Cloud TPU using the XRT client, set the XRT_TPU_CONFIG environment variable:

  ```Shell
  export XRT_TPU_CONFIG="tpu_worker;0;<IP of the TPU node>:8470"
  ```

Note that the IP of the TPU node can change if the TPU node is reset. If _PyTorch_
seem to hang at startup, verify that the IP of your TPU node is still the same of
the one you have configured.

If you are planning to be building from source and hence using the latest _PyTorch/TPU_ code base,
it is suggested for you to select the _Nightly_ builds when you create a Cloud TPU instance.

Then run `test/run_tests.sh` and `test/cpp/run_tests.sh` to verify the setup is working.


[![CircleCI](https://circleci.com/gh/pytorch/xla.svg?style=svg)](https://circleci.com/gh/pytorch/xla)

# Debugging

Sometimes bad things happen and a deeper look into the _PyTorch/TPU_ stack is necessary.
In order to do that, _PyTorch/TPU_ has a series of environment variables and function calls
which can help understading its internal behavior.

Note that the infromation in this section is subject to be removed in future releases of
the _PyTorch/TPU_ software, since many of them are peculiar to a given internal implementation
which might change.

The _PyTorch/TPU_ stack keeps a series of metrics and counters during its execution, and
the following API returns a string representation of them:

```Python
torch_xla._XLAC._xla_metrics_report()
```

Printing out that information can help during the debug phases and while reporting issues.

The information included within the metrics report include things like how many time we
issue _XLA_ compilations, how long they take, how many times we execute, for how long,
how many device data handles we create/destroy, etc...
These information is reported in terms of percentiles of the samples.
An example is:

```
Metric: CompileTime
  TotalSamples: 202
  Counter: 06m09s401ms746.001us
  ValueRate: 778ms572.062us / second
  Rate: 0.425201 / second
  Percentiles: 1%=001ms32.778us; 5%=001ms61.283us; 10%=001ms79.236us; 20%=001ms110.973us; 50%=001ms228.773us; 80%=001ms339.183us; 90%=001ms434.305us; 95%=002ms921.063us; 99%=21s102ms853.173us
```

The _PyTorch/TPU_ stack also has counters, which are named integer variables tracks
internal software status.
Example:

```
Counter: CachedSyncTensors
  Value: 395
```

Counters are also useful to understand which operations the _PyTorch/TPU_ stack is routing
back to the CPU engine of _PyTorch_.
Things which looks like a _C++_ namespace are part of this category:

```
Counter: aten::nonzero
  Value: 33
```

There are also a number of environment variables which control the behavior of the _PyTorch/TPU_
software stack.
Setting such variables will cause different degrees of performance degradation, so they should
only be enabled for debugging.

* ```XLA_IR_DEBUG```: Enables the _Python_ stack trace to be catpured where creating IR nodes,
  hence allowing to understand which _PyTorch_ operation was responsible of generating such IR.

* ```XLA_HLO_DEBUG```: Enables the _Python_ stack frame captured when _XLA_IR_DEBUG_ is active,
  to be propagated to the _XLA_ _HLO_ metadata.

* ```XLA_SAVE_TENSORS_FILE```: The path to a file which will be used to dump the IR graphs during
  execution. Note that the file can become really big if the option is left enabled and the
  _PyTorch_ program let run for long time. The graphs are appended to the file, so to have a clean
  sheet from run to run, the file should be explicitly removed.

* ```XLA_SAVE_TENSORS_FMT```: The format of the graphs stored within the _XLA_SAVE_TENSORS_FILE_
  file. Can be ```text``` (the default), ```dot``` (the _Graphviz_ format) or ```hlo```.

* ```XLA_METRICS_FILE```: If set, the path to a local file where the internal metrics will be
  saved at every step. Metrics will be appended to the file, if already existing.

* ```GET_TENSORS_OPBYOP```: Enables pure _OpByOp_ dispatch. The _PyTorch/TPU_ software tries to
  fuse together many _PyTorch_ operations into a single computation graph, but sometimes, either
  for debugging, or in case the _PyTorch_ code have a very dynamic nature (in shapes or graph
  terms), it is better to force the execution in _OpByOp_ mode (every IR node is lowered into
  a separate _XLA_ computation, and chain-executed). This environment variable, if set to 1,
  enables _OpByOp_ during the "get tensors" operation (the operation used by _PyTorch/TPU_ to
  fetch intermediate values back from the _TPU_ device into _PyTorch_ CPU tensors).

* ```SYNC_TENSORS_OPBYOP```: The same as _GET_TENSORS_OPBYOP_ but for "sync tensors" operation
  (the operation used at the end of a step, to flush pending IR computations and materialize
  them into _TPU_ device data).

* ```XLA_SYNC_WAIT```: Forces the XLA tensor sync operation to wait for its completion, before
  moving to the next step.

* ```XLA_USE_BF16```: If set to 1, tranforms all the _PyTorch_ _Float_ values into _BiFloat16_
  when sending to the _TPU_ device.

* ```XLA_USE_32BIT_LONG```: If set to 1, maps _PyTorch_ _Long_ types to _XLA_ 32bit type.
  On the versions of the TPU HW at the time of writing, 64bit integer computations are
  expensive, so setting this flag might help. It should be verified by the user that truncating
  to 32bit values is a valid operation according to the use of _PyTorch_ _Long_ values in it.

## Retrieving Stack Traces

In the event that the _PyTorch_ process is hanging, it might be useful to include the stack
traces together with the _Github_ issue.

First thing is to find out which PID the _PyTorch_ process is associated with. Using the ```ps```
command it is possible to find that information. It will be a _python_ process running your
main _python_ file.

In order to allow _GDB_ to attach a user process the following command should be run as root:

```Shell
echo 0 > /proc/sys/kernel/yama/ptrace_scope
```

The above command remains active until the machine is rebooted.

The, given the PID, it is possible to grab the stack traces with the following command:

```Shell
./scripts/dump_stacks.py PID > /tmp/stack-traces.log
```


## Communication

We use github issues to communicate with users and open source contributors. Please file an issue for questions, bug reports, feature requests, install issues, RFCs, thoughts, etc.

## Contributing

We appreciate all contributions. If you are planning to contribute bug fix for an open issue, please comment on the thread and we're happy to provide any guidance. You are very welcome to pick issues from `good first issue` and `help wanted` labels.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us.
Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

Please refer to [contribution guide](CONTRIBUTING.md) for detailed guidelines to submit PRs.
