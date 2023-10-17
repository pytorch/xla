# Troubleshooting

Note that the information in this section is subject to be removed in future releases of the _PyTorch/XLA_ software,
since many of them are peculiar to a given internal implementation which might change.

## Sanity Check
Before performing any in depth debugging, we want to do a sanity check on the installed PyTorch/XLA.

### Check PyTorch/XLA Version
PyTorch and PyTorch/XLA version should match. Check out our [README](https://github.com/pytorch/xla#getting-started) for more detials on versions available.
```
vm:~$ python
>>> import torch
>>> import torch_xla
>>> print(torch.__version__)
2.1.0+cu121
>>> print(torch_xla.__version__)
2.1.0
```

### Perform A Simple Calculation
```
vm:~$ export PJRT_DEVICE=TPU
vm:~$ python3
>>> import torch
>>> import torch_xla.core.xla_model as xm
>>> t1 = torch.tensor(100, device=xm.xla_device())
>>> t2 = torch.tensor(200, device=xm.xla_device())
>>> print(t1 + t2)
tensor(300, device='xla:0')
```

### Run Resnet With Fake Data
For nightly
```
vm:~$ git clone https://github.com/pytorch/xla.git
vm:~$ python xla/test/test_train_mp_imagenet.py --fake_data
```

For release version `x.y`, you want to use the branch `rx.y`. For example if you installed 2.1 release, you should do
```
vm:~$ git clone --branch r2.1 https://github.com/pytorch/xla.git
vm:~$ python xla/test/test_train_mp_imagenet.py --fake_data
```

If you can get the resnet to run we can conclude that torch_xla is installed correctly. 


## Performance Debugging

To diagnose performance issues, we can use the execution metrics and counters provided by _PyTorch/XLA_
The **first thing** to check when model is slow is to generate a metrics report.

Metrics report is extremely helpful in diagnosing issues. Please try to include it in your bug
report sent to us if you have it.

## Perform A Auto-Metrics Analysis

We provide ways to automatically analyze the metrics report and provide a summary. Simply run your workload with `PT_XLA_DEBUG=1`. Some example output would be

```
pt-xla-profiler: CompileTime too frequent: 21 counts during 11 steps
pt-xla-profiler: TransferFromServerTime too frequent: 11 counts during 11 steps
pt-xla-profiler: Op(s) not lowered: aten::_ctc_loss, aten::_ctc_loss_backward,  Please open a GitHub issue with the above op lowering requests.
pt-xla-profiler: CompileTime too frequent: 23 counts during 12 steps
pt-xla-profiler: TransferFromServerTime too frequent: 12 counts during 12 steps
```

Following section will explain how to get and understand a more detail metrics report.

## Get A Metrics Report

Put the following line in your program to generate a report:

```Python
import torch_xla.debug.metrics as met

# For short report that only contains a few key metrics.
print(met.short_metrics_report())
# For full report that includes all metrics.
print(met.metrics_report())
```

## Understand The Metrics Report

The report includes things like:
- how many time we issue _XLA_ compilations and time spent on issuing.
- how many times we execute and time spent on execution
- how many device data handles we create/destroy etc.

This information is reported in terms of percentiles of the samples. An example is:

```
Metric: CompileTime
  TotalSamples: 202
  Counter: 06m09s401ms746.001us
  ValueRate: 778ms572.062us / second
  Rate: 0.425201 / second
  Percentiles: 1%=001ms32.778us; 5%=001ms61.283us; 10%=001ms79.236us; 20%=001ms110.973us; 50%=001ms228.773us; 80%=001ms339.183us; 90%=001ms434.305us; 95%=002ms921.063us; 99%=21s102ms853.173us
```

We also provide counters, which are named integer variables which track internal software status. For example:

```
Counter: CachedSyncTensors
  Value: 395
```

In this report, any counter that starts with `aten::`
indicates a context switch between the XLA device and CPU, which can be a
potential performance optimization area in the model code.

Counters are useful to understand which operations are routed back to the CPU engine of _PyTorch_.
They are fully qualified with their C++ namespace:

```
Counter: aten::nonzero
  Value: 33
```

If you see `aten::` ops other than `nonzero` and `_local_scalar_dense`, that usually means a missing
lowering in PyTorch/XLA. Feel free to open a feature request for it on [GitHub issues](https://github.com/pytorch/xla/issues).

## Clar The Metrics Report
If you want to clear the metrics between steps/epochs, you can use
```Python
import torch_xla.debug.metrics as met

met.clear_all()
```

## Performance Profiling
To profile your workload in depth to understand bottlenecks please check the following resources:
* [Official tutorial](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm)
* [Colab notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/pytorch-xla-profiling-colab.ipynb)
* [Sample MNIST training script with profiling](https://github.com/pytorch/xla/blob/master/test/test_profile_mp_mnist.py)
* [Utility script for capturing performance profiles](https://github.com/pytorch/xla/blob/master/scripts/capture_profile.py)

## Known Performance Caveats

PyTorch/XLA behaves semantically like regular PyTorch and XLA tensors share the full tensor interface with CPU & GPU tensors.
However, constraints in XLA/hardware and the lazy evaluation model suggest certain patterns might result in bad performance.

If your model shows bad performance, keep in mind the following caveats:

1.  **XLA/TPU yield degraded performance with too many recompilations.**

    XLA compilation is expensive. PyTorch/XLA automatically recompiles the graph every time new shapes are encountered.
    Usually models should stabilize within a few steps and you can see huge speedup for the rest of training.

    In order to avoid recompilations, not only must shapes be constant, but computations across XLA devices in all hosts should also be constant.

    _Possible sources_:
    * Direct or indirect uses of `nonzero` introduce dynamic shapes; for example, masked indexing `base[index]` where `index` is a mask tensor.
    * Loops with a different number of iterations between steps can result in different execution graphs, thus require recompilations.

    _Solution_:
    * Tensor shapes should be the same between iterations, or a low number of shape variations should be used.
    * Pad tensors to fixed sizes when possible.

1.  **Certain operations don't have native translations to XLA.**

    For these operations PyTorch/XLA automatically transfers to the CPU memory, evaluates on CPU, and transfers the result back to the XLA device.
    Doing too many such operations during the training step can lead to significant slowdowns.

    _Possible sources_:

    - The `item()` operation explicitly asks to evaluate the result. Don't use it unless it's necessary.

    _Solution_:

    - For most ops we can lower them to XLA to fix it. Checkout [metrics report section](#metrics-report) to find out the missing ops and open a feature request on [GitHub](https://github.com/pytorch/xla/issues).
    - Even when a PyTorch tensor is known as a scalar, avoid using `tensor.item()`. Keep it as a tensor and use tensor operations on it.
    - Use `torch.where` to substitute control flow when applicable.
      E.g. The control flow with `item()` used in [clip_grad_norm_](https://github.com/pytorch/pytorch/blob/de19eeee99a2a282fc441f637b23d8e50c75ecd1/torch/nn/utils/clip_grad.py#L33) is problematic and impacts performance, so we have [patched](https://github.com/pytorch/xla/blob/master/torch_patches/X10-clip_grad.diff) `clip_grad_norm_` by calling `torch.where` instead, which gives us a dramatic performance improvement.
      ```python
      ...
      else:
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        for p in parameters:
          param_norm = p.grad.data.norm(norm_type) ** norm_type
          total_norm.add_(param_norm)
        total_norm = (total_norm ** (1. / norm_type))
      clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
      for p in parameters:
        p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))
      ```

1. **Iterators in `torch_xla.distributed.data_parallel` may drop the last few batches in the input iterator.**

   This is to make sure we do the same amount of work on all XLA devices.

   _Solution_:

   * When dataset is small, and there are too few steps, this may result in a no-op epoch. Therefore, it is better to use
   small batch sizes in those cases.

## XLA Tensor Quirks

1. **XLA tensor internals are opaque.** XLA tensors always appear to be
contiguous and without storage. Networks should not try to check the strides
of XLA tensors.

1. **XLA tensors should be moved to the CPU before saving them.** Saving
XLA tensors directly causes them to be loaded back on the device(s) they were
saved from. If a device is unavailable at load time then the load will fail.
Moving XLA tensors to the CPU before saving them lets you decide which
device(s) to put the loaded tensors on. This is necessary if you want to
load the tensors on a machine without XLA devices. Care should be taken
moving the XLA tensors to the CPU before saving them, however, as moving
tensors across device types does not preserve view relationships. Instead,
views should be reconstructed as necessary after the tensors are loaded.

1. **Copying an XLA Tensor with Python's copy.copy returns a deep copy, not a
shallow copy.** Use a view of an XLA tensor to get a shallow copy of it.

1. **Handling shared weights.** Modules can share weights by setting the
Parameters of one module to another. This "tying" of module weights should
be done **AFTER** the modules are moved to an XLA device. Otherwise two
independent copies of the shared tensor will be made on the XLA device.

## More Debugging Tools

We don't expect users to use tools in this section to debug their models. But we might ask for
them when you submit a bug report since they provide additional information that metrics report
doesn't have.

### Environment Variables

There are also a number of environment variables which control the behavior of the _PyTorch/XLA_
software stack.

Setting such variables will cause different degrees of performance degradation, so they should
only be enabled for debugging.

* ```XLA_IR_DEBUG```: Enables the _Python_ stack trace to be captured where creating IR nodes,
  hence allowing to understand which _PyTorch_ operation was responsible for generating the IR.

* ```XLA_HLO_DEBUG```: Enables the _Python_ stack frame captured when _XLA_IR_DEBUG_ is active,
  to be propagated to the _XLA_ _HLO_ metadata.

* ```XLA_SAVE_TENSORS_FILE```: The path to a file which will be used to dump the IR graphs during
  execution. Note that the file can become really big if the option is left enabled and the
  _PyTorch_ program let run for long time. The graphs are appended to the file, so to have a clean
  sheet from run to run, the file should be explicitly removed.

* ```XLA_SAVE_TENSORS_FMT```: The format of the graphs stored within the _XLA_SAVE_TENSORS_FILE_
  file. Can be ```text``` (the default), ```dot``` (the _Graphviz_ format) or ```hlo```.

* ```XLA_FLAGS=--xla_dump_to```: If set to ```=/tmp/dir_name```, XLA compiler will dump the unoptimized and optimzed HLO per compilation.

* ```XLA_METRICS_FILE```: If set, the path to a local file where the internal metrics will be
  saved at every step. Metrics will be appended to the file, if already existing.

* ```XLA_SAVE_HLO_FILE```: If set, the path to a local file where, in case of compilation/execution
  error, the offending HLO graph will be saved.

* ```XLA_SYNC_WAIT```: Forces the XLA tensor sync operation to wait for its completion, before
  moving to the next step.

* ```XLA_USE_BF16```: If set to 1, transforms all the _PyTorch_ _Float_ values into _BiFloat16_
  when sending to the _TPU_ device. Note that when using `XLA_USE_BF16=1` tensor arithmetic will
  be done in reduced precision and so tensors will not be accurate if accumulated over time.
  For example:

  ```
  # In reduced bfloat16 precision
  >>> torch.tensor(4096, dtype=torch.bfloat16) + torch.tensor(1, dtype=torch.bfloat16)
  tensor(4096., dtype=torch.bfloat16)
  # Whereas in full float32 precision
  >>> torch.tensor(4096) + torch.tensor(1)
  tensor(4097)
  ```
  So to get accurate metrics such as average loss value over many steps, use manual mixed
  precision where metrics stay in FP32.

* ```XLA_USE_F16```: If set to 1, transforms all the _PyTorch_ _Float_ values into _Float16_
  (_PyTorch_ _Half_ type) when sending to devices which supports them.

* ```XLA_USE_32BIT_LONG```: If set to 1, maps _PyTorch_ _Long_ types to _XLA_ 32bit type.
  On the versions of the TPU HW at the time of writing, 64bit integer computations are
  expensive, so setting this flag might help. It should be verified by the user that truncating
  to 32bit values is a valid operation according to the use of _PyTorch_ _Long_ values in it.

* ```TF_CPP_LOG_THREAD_ID```: If set to 1, the TF logs will show the thread ID
  helping with debugging multithreaded processes.

* ```TF_CPP_VMODULE```: Environment variable used for TF VLOGs and takes the
  form of `TF_CPP_VMODULE=name=value,...`. Note that for VLOGs you must set
  `TF_CPP_MIN_LOG_LEVEL=0`. For PyTorch/XLA using a configuration like
  `TF_CPP_VMODULE=tensor=5` would enable logging such as:

  ```
  2019-10-03 17:23:56.419040: I   27891 torch_xla/csrc/tensor.cpp:1104]
  Executing IR graph hash 4211381954965020633 on device TPU:3 done!
  2019-10-03 17:23:56.419448: I   27890 torch_xla/csrc/tensor.cpp:1104]
  Executing IR graph hash 15483856951158150605 on device TPU:5 done!
  2019-10-03 17:23:56.419539: I   27896 torch_xla/csrc/tensor.cpp:1104]
  Executing IR graph hash 4211381954965020633 on device TPU:4 done!
  ...
  ```

* ```TF_CPP_MIN_LOG_LEVEL```: Level to print messages for. `TF_CPP_MIN_LOG_LEVEL=0` will turn
  on INFO logging, `TF_CPP_MIN_LOG_LEVEL=1` WARNING and so on. Our PyTorch/XLA `TF_VLOG` uses
  `tensorflow::INFO` level by default so to see VLOGs set `TF_CPP_MIN_LOG_LEVEL=0`.

* ```XLA_DUMP_HLO_GRAPH```: If set to `=1` in case of a compilation or execution error the
  offending HLO graph will be dumped as part of the runtime error raised by `xla_util.cc`.

