# Torchbench and PyTorch/XLA

[Torchbench][1] is a collection of open-source benchmarks used for evaluating PyTorch
performance.

PyTorch has [its own set of scripts][2] for running these benchmarks. They are used on
PyTorch's CI as well as their own [compiler's HUD web-page][3], which frequently checks
the performance of PyTorch dynamo (`torch.compile`) backends, such as inductor, inductor
with CUDAGraphs, etc.

In that sense, PyTorch/XLA also has its own set of benchmarking scripts that are used for
evaluating PyTorch/XLA performance, and compare it with the performance of PyTorch. The
scripts attempt to replicate the numbers in the PyTorch HUD, aiming towards a fair
comparison against PyTorch's main backend: inductor.

## Prerequisites

Make sure you have installed all of:

- [PyTorch][5]
- [PyTorch/XLA][6]
- [Torchbench][1]

Note that PyTorch/XLA follows [PyTorch's pin][7] for Torchbench. So, your mileage may vary
on other Torchbench versions. In order to make sure everything works as expected, we
recommend matching PyTorch and PyTorch/XLA by their commit date. Then, using whatever is
the Torchbench pin for that PyTorch version.

For the rest of this document, assume the following directory hierarchy:

```
- pytorch (current directory)
    |- ...
    |- xla (PyTorch/XLA root directory)
```

## Getting Started

The main entry-point for executing Torchbench with PyTorch/XLA benchmarking scripts is:
[_experiment_runner.py_][4]. See below the simplest command for running Torchbench.


```bash
python xla/benchmarks/experiment_runner.py \
    --suite-name=torchbench \
    --accelerator=cuda
```

It will run the whole Torchbench collection, with all possible configurations, repeating
10 times a 1-iteration run. For example, assume that Torchbench is composed of only 2
benchmarks (currently, there are 103): `llama` and `dlrm`. Then, the command will run 16
times (8 for each benchmark):

```
model: llama
 1. xla: None, dynamo: None,     test: eval
 2. xla: None, dynamo: None,     test: train
 3. xla: None, dynamo: inductor, test: eval
 4. xla: None, dynamo: inductor, test: train
 5. xla: PJRT, dynamo: None,     test: eval
 6. xla: PJRT, dynamo: None,     test: train
 7. xla: PJRT, dynamo: openxla,  test: eval
 8. xla: PJRT, dynamo: openxla,  test: train

model: dlrm
 9. xla: None, dynamo: None,     test: eval
10. xla: None, dynamo: None,     test: train
11. xla: None, dynamo: inductor, test: eval
12. xla: None, dynamo: inductor, test: train
13. xla: PJRT, dynamo: None,     test: eval
14. xla: PJRT, dynamo: None,     test: train
15. xla: PJRT, dynamo: openxla,  test: eval
16. xla: PJRT, dynamo: openxla,  test: train
```

Each of `xla:`, `dynamo:`, and `test:` are parameters to the _experiment_runner.py_ script
for filtering the desired experiment configurations. You can specify each of them as many
times as necessary, e.g. `--dynamo inductor --dynamo openxla`, filtering-out experiments
with configuration `dynamo: None`. Additionally, you can filter models by specifying the
`--filter` parameter as many times as you want. For example:

```bash
python xla/benchmarks/experiment_runner.py \
    --suite-name=torchbench \
    --accelerator=cuda \
    --xla=PJRT \
    --dynamo=openxla --dynamo=None \
    --test=eval \
    --filter=llama
```

It will run only `llama` inference (`--test=eval`) using PJRT with and without dynamo. Or,
more verbosely:

```
model: llama
1. xla: PJRT, dynamo: None,     test: eval
2. xla: PJRT, dynamo: openxla,  test: eval
```

You can also modify the number of repetitions (`--repeat`) as well as the number of
iterations (`--iterations-per-run`) to be run in each repetition. For a more detailed list
of the available command-line arguments, run _experiment_runner.py_ with the `--help`
argument:

```bash
python xla/benchmarks/experiment_runner.py --help
```

Finally, the benchmarking script will output the file _output/results.jsonl_, where each
line will contain: (i) the experiment information, e.g. model name, xla, dynamo, test,
etc.; and (ii) collected metrics, e.g. total running time, etc. A line in that file will
look like the following:

```
{
  "model": {
    "suite_name": "torchbench",
    "model_name": "llama"
  },
  "experiment": {
    "accelerator": "cuda",
    "accelerator_model": "NVIDIA ...",
    "xla": "PJRT",
    "xla_flags": null,
    "dynamo": "openxla",
    "torch_xla2": null,
    "keep_model_data_on_cuda": false,
    "test": "eval",
    "batch_size": 1
  },
  "repeat": 5,
  "iterations_per_run": 1,
  "metrics": {
    "total_time": [...],
    ...
  },
  "timestamp": ...,
  "verification_code": "VERIFIER_SKIPPED"
}
```

## Troubleshooting and Debugging

### Run the Verifier

First things of, make sure that the benchmark you are trying to run actually meets [the
accuracy we expect][#model-verification] by passing the `--verify` flag to the
_experiment_runner.py_ script. If the benchmark does passes the check, you should see the
following in your _results.jsonl_ file:

```
{
  "model": {
    "suite_name": "torchbench",
    "model_name": "llama"
  },
  ...
  "verification_code": "PASS"
}
```

With this result in hands, if the `verification_code` is something other than `PASS` it
means either that the result (prediction/gradients) is wrong, or that we cannot tell, for
some reason. More specifically:

| Flag                         | Troubleshoot                                                  |
|:----------------------------:|---------------------------------------------------------------|
| `FAIL`                       | Result is wrong                                               |
| `EAGER_FAILED`               | This benchmark should not be running (it is failing on eager) |
| `EXCEPTION_RAISED`           | There may be something wrong with the verifier                |
| `NONDETERMINISTIC_EAGER_RUN` | Eager is not deterministic, so cannot compare the results     |
| `VERIFIER_SKIPPED`           | We should skip the verifier for this benchmark                |
| `VERIFIER_DIDNT_RUN`         | There was some uncaught error with the verifier process       |

Please, open an issue if you see any of the codes in the table above, except for
`FAIL`. If you see `FAIL`, but did not introduce any change to any code, please, open an
issue. Otherwise, it means that your change likely introduced some incorrect behavior.


## Experiment Result

The benchmarking scripts will store the resulting artifacts in a directory called `output`
(default) or another one specified by the parameter `--output-dirname`.

### Experiments and Metrics

When running _experiment_runner.py_ script, a [JSONL][8] file called _results.jsonl_ will
be created (if it doesn't already exists), where each JSON line corresponds to an
experiment instance. Each line contains: (i) experiment information; and (ii) collected
metrics.

#### Collected Metrics

By default, the benchmarking scripts will collect a 2 metrics:

1. `total_time`: time taken for running all iterations
2. `per_iter_time`: average iteration time

If the experiment runner actually uses one of PyTorch/XLA backends, the scripts will
additionally collect 6 new metrics:

1. `trace_total_time`: sum of the tracing time for all iterations
2. `trace_per_iter_time`: average tracing time
3. `xla_CompileTime_time_s`: sum of the compilation time for all generated graphs
4. `xla_CompileTime_number`: number of graphs compiled
5. `xla_ExecutionTime_time_s`: sum of the execution time for all compiled graphs
6. `xla_ExecutionTime_number`: number of times that a compiled graph was executed

#### Optional Metrics

**Dynamo Counters:** the flag `--collect-dynamo-counters` appends the PyTorch dynamo
counters to the metrics information. For example, number of successfully traced frames,
number of FX graphs compiled successfully by AOTAutograd, inductor related information,
etc.

**Profiler Metrics for CPU and CUDA:** the flag `--profile-cuda-cpu` collects CPU and CUDA
profiling information about the execution. Specifically, it collects 4 of them:

1. `total_cpu_time_s`: total CPU time take for running all iterations
2. `per_iter_cpu_time_s`: average CPU time taken on each iteration
3. `total_cuda_time_s`: total CUDA time take for running all iterations
4. `per_iter_cuda_time_s`: average CPU time taken on each iteration

**Profiler Metrics for CPU and CUDA for unlowered operations:** the flag
`--profile-cuda-cpu-individual-ops` collects profiling information for operations that
went through the fallback path, i.e. those for which PyTorch/XLA does not have a
lowering. For each of those operations, the scripts collect 5 new metrics:

1. `self_cpu_time_s`: CPU time spent in that operation
2. `total_cpu_time_s`: CPU time spent running that function, including other dependent
   operations
3. `self_cuda_time_s`: CUDA time spent in that operation
4. `total_cuda_time_s`: CUDA time spent running that function, including other dependent
   operations
5. `num_of_calls`: number of times that operation was called

### Dumping Benchmark-specific Data

Besides the collected metrics, the benchmarking scripts have the option for dumping
information specific to each benchmark. These are stored in a newly created directory for
each benchmark, under the output directory. The descriptions below assume all paths are
relative to the benchmark-specific directory.

There is information that is collected for each repetition (controlled by `--repeat`
parameter) of a benchmark. Those are stored in separate directories, one for each
repetition. For better clarity, we shall refer to those flags by marking them with the
_"repeat-specific"_ tag.

**HLO Modules:** the flag `--dump-hlo` sets the environment variable `XLA_FLAGS`, dumping
XLA compiled HLO modules into the _"hlo"_ directory.

**Dynamo Counters:** (repeat-specific) the flag `--dump-dynamo-counters` is similar to the
already mentioned `--collect-dynamo-counters` flag. However, instead of collecting them
and storing in a JSON line inside _results.jsonl_, setting this flag will dump the dynamo
counter dictionary into a new file.

**PyTorch Profiles and Trace:** (repeat-specific) the flag `--dump-pytorch-profiles`
creates 2 new files. For more detailed information, check out [this post][9] explaining
PyTorch profiling:

1. Chrome trace: same as calling `profiler.export_chrome_trace`
2. Profile: same as calling `profiler.key_averages().table(...)`, a summarized table with
   the timing averages

**PyTorch/XLA Metrics:** (repeat-specific) the flag `--dump-pytorch-xla-metrics` creates a
new file, dumping PyTorch/XLA metrics, such as graph compiling and execution information.

**Intermediate Representations (IR):** the parameter `--save-ir` allows the selection of
one of the used IRs: `hlo` (similar to `--dump-hlo` above), `stablehlo`, and `text`
(PyTorch lazy IR).

## Model Verification

In order to validate the execution of the benchmarks, it is possible to run an accuracy
check on them by specifying the flag `--verify`. For consistency reasons, we run a similar
version of the accuracy test run in the PyTorch HUD, which, at its core, uses [the `same`
function][10]. In summary, it tries to check the accuracy of the inference
prediction/training gradients by running one of these three methods: (i) cosine
similarity; (ii) `torch.allclose`; and (iii) root mean squared error (RMSE).

As the result of the verification, the benchmarking scripts will change the
`verification_code` field of the corresponding line of the _results.jsonl_ file with one
of the following:

| Flag                         | Short Description                                        |
|:----------------------------:|----------------------------------------------------------|
| `PASS`                       | Verification passed the accuracy check                   |
| `FAIL`                       | Verification failed the accuracy check                   |
| `EAGER_FAILED`               | Eager execution failed                                   |
| `EXCEPTION_RAISED`           | An exception was raised when running the verifier        |
| `NONDETERMINISTIC_EAGER_RUN` | Two eager runs of the benchmark have mismatching results |
| `VERIFIER_SKIPPED`           | Verification was skipped                                 |
| `VERIFIER_DIDNT_RUN`         | Verification did not run due to an unexpected reason     |


[1]: https://github.com/pytorch/benchmark
[2]: https://github.com/pytorch/pytorch/blob/main/benchmarks/dynamo/torchbench.py
[3]: https://hud.pytorch.org/benchmark/compilers
[4]: https://github.com/pytorch/xla/blob/master/benchmarks/experiment_runner.py
[5]: https://github.com/pytorch/pytorch
[6]: https://github.com/pytorch/xla
[7]: https://github.com/pytorch/pytorch/blob/main/.github/ci_commit_pins/torchbench.txt
[8]: https://jsonlines.org/
[9]: https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
[10]: https://github.com/pytorch/pytorch/blob/a4e9a1c90b74572b48f2eedf1e931c18713c1781/torch/_dynamo/utils.py#L1616
