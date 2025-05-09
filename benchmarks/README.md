# Benchmarking

The two main benchmarking scripts are
  - `experiment_runner.py` to run benchmark experiments, and
  - `result_analyzer.py` to aggregate the benchmark result in CSV form.


## Patching mismatched batch sizes

Sometimes batch sizes for inference might differ between Inductor, and XLA.
This stems from the fact that we pass in an XLA device string to the TorchBench
modelling code, instead of a raw CUDA string, and the path to correctly
fetch the accelerator underneath is not covered. To fix this apply a patch:

```
git apply benchmarks/patches/mismatched_batch_size.patch
```

And replace the `current_device_name` with your actual accelerator name.

## Reducing benchmark noise

It is important to keep the benchmark runs safe from external effects
to reduce noise. Do the following:

Sets the CPU statically to the highest tuneable frequency.
Prevent energy saving features to kick in.

```sudo cpupower frequency-set --governor performance```

Lock GPU clocks to lower frequency to reduce the chance of thermal throttling. Choose
FREQ based on your GPU info. To find out clock frequency on your device run:
`nvidia-smi -q -d CLOCK`, and look for Graphics/SM in Max Clocks section.
Setting the clock a couple hundrend MHz below, or ~80% of max
will most likely prevent thermal throttling effects.

```FREQ=... nvidia-smi --lock-gpu-clocks=$FREQ,$FREQ```

Disable autoboost selecting clock rate based on thermal, and power budget effects.
```CUDA_AUTO_BOOST=0```

## Experiment runner

Run the `experiment_runner.py` from the `pytorch` directory, which should be the
parent of the `xla` directory.

The following example runs the alexnet benchmark on GPU through the
Pytorch/XLA-dynamo path and through the Inductor-dynamo with 5 repetitions each.
The results will be stored in a json file (eg results.jsonl) in `experiment_results`.

```
cd pytorch
python xla/benchmarks/experiment_runner.py  \
    --dynamo=openxla --dynamo=inductor      \
    --xla=PJRT --xla=None                   \
    --test=eval --test=train                \
    --suite-name=torchbench                 \
    --accelerator=cuda                      \
    --output-dirname=experiment_results     \
    --repeat=5                              \
    --print-subprocess                      \
    --no-resume                             \
    --filter="^alexnet$"
```

You can change the flags to add the configurations you are interested in. The
`experiment_runner.py` will expand the options to all supported configurations.
For example, in the case above, it will consider all the possible combinations
among the flags `--dynamo`, `--xla`, and `--test`, 4 of which are supported:

  - `dynamo=openxla`, `xla=PJRT`, `test=eval`
  - `dynamo=openxla`, `xla=PJRT`, `test=train`
  - `dynamo=inductor`, `xla=None`, `test=eval`
  - `dynamo=inductor`, `xla=None`, `test=train`


## Run benchmarking for a single configuration

The section `Experiment runner` above shows how to run the benchmarking script for a combination of configurations. For each configuration,
the script starts a process and run the benchmarking. This section shows how to run the benchmarking for a single configuration without spawning new processes.

```
cd pytorch
python xla/benchmarks/experiment_runner.py \
    --suite-name=torchbench \
    --progress-bar  \
    --model-config='{"model_name":"BERT_pytorch"}' \
    --experiment-config='{"accelerator":"cuda","xla":"PJRT","xla_flags":null,"dynamo":"openxla","torch_xla2":null,"test":"train","keep_model_data_on_cuda":false,"enable_functionalization":false}' \
    --repeat 1
```


## Verification module

Verification flag, enabled by running the experiment runner script with `--verify`
calculates the mean relative error of the model's output against the eager run
of the very same model. If the difference is greater than predefined threshold (e.g. 2%)
it will report the `FAIL` status code in the output file of the benchmarking one, and `PASS`
if it is not. Additional verification codes can be present if the verification fails
due to various issues (e.g. unsupported output shape, failed eager run). The verification
works only for inference now.

```
cd pytorch
PJRT_DEVICE=CUDA python3 new_xla/benchmarks/experiment_runner.py \
    --xla=PJRT \
    --dynamo=openxla \
    --test=eval \
    --filter=BERT_pytorch$ \
    --suite-name=torchbench \
    --accelerator=cuda \
    --progress-bar \
    --output-dirname=/tmp/output \
    --repeat=2 \
    --print-subprocess \
    --no-resume \
    --verify

cat /tmp/output/results.jsonl
{[...] "verification_code": "PASS", "verification_mean_rel_error": 0.007134194485843182}
```

## Microbenchmarks

In `bench.py` there is a common infrastructure to measure things without
CPU, and CUDA synchronisation overhead. `matmul_benchmark.py` is the microbenchmark
which utilizes this  infra to perform a simple squared matrix multiplication for
PT/XLA, and compare it against some basline.

## Result analyzer

Run the `result_analyzer.py` from the `pytorch` directory, which should be the
parent of the `xla` directory.

The following example analyzes the results (eg results.jsonl) generated by the above invocation of
`experiment_runner.py`. So make sure to use consistent `--output-dirname` parameter. The aggregates are saved in CSV format in
`experiment_results/metric_report.csv`.

```
cd pytorch
python xla/benchmarks/result_analyzer.py --output-dirname=experiment_results
```

## Aggregating results

Aggregate reports can be generated directly from the output JSONL files
(i.e., skipping `result_analyzer.py` altogether) with the `aggregate.py` script.
The script compares Pytorch/XLA performance numbers against Inductor numbers.
Because Inductor's performance also changes over time, the script takes
the oldest Inductor performance numbers present in the JSONL files (as
determined by the records' timestamp) as the baseline for each benchmark.

Sample runs and sample output:

- Note: the numbers themselves are irrelevant; do not pay attention to them.
- Note 2: we only have experiments from a single day so far, which makes the
  sample plots less interesting.
- Note 3: we are using ASCII output here just to avoid checking in PNG files.

```
$ python3 aggregate.py --accelerator=v100 --test=inference --format=png --report=histogram /tmp/test/*.jsonl

                Histogram of Speedup over Oldest Benchmarked Inductor
     1.2 +------------------------------------------------------------------+
         |      +       +      +       +   D  +       +      +       +      |
       1 |-+                               C           Inductor p95    A  +-|
         |                                             Inductor p50    B    |
     0.8 |-+                                            Inductor p5    C  +-|
         |                                           PytorchXLA p95    D    |
     0.6 |-+                                         PytorchXLA p50    E  +-|
         |                                            PytorchXLA p5    F    |
         |                                 E                                |
     0.4 |-+                                                              +-|
         |                                                                  |
     0.2 |-+                                                              +-|
         |      +       +      +       +   F  +       +      +       +      |
       0 +------------------------------------------------------------------+
        2000   2005    2010   2015    2020   2025    2030   2035    2040   2045
                                        Date

$ python3 aggregate.py --accelerator=v100 --test=inference --format=png --report=speedup /tmp/test/*.jsonl

        Geomean Speedup Over Oldest Benchmarked Inductor
       1 +----------------------------------------------+
         |    +    +     +    +    +    +     +    +    |
     0.9 |-+                           Inductor    A  +-|
         |                           PytorchXLA    B    |
     0.8 |-+                                          +-|
         |                                              |
     0.7 |-+                                          +-|
         |                                              |
         |                                              |
     0.6 |-+                                          +-|
         |                                              |
     0.5 |-+                                          +-|
         |    +    +     +    +  B +    +     +    +    |
     0.4 +----------------------------------------------+
        2000 2005 2010  2015 2020 2025 2030  2035 2040 2045
                              Date
$ python3 aggregate.py --accelerator=v100 --test=inference --format=png --report=latest /tmp/test/*.jsonl

Speedup Over Oldest Benchmarked Inductor as of 2023-11-11
     1.8 +----------------------------------------------+
     1.6 |-+  +    +     +    +    BB   +     +    +  +-|
         |                             Inductor    A    |
     1.4 |-+                         PytorchXLA    B  +-|
     1.2 |-+                                          +-|
         |                       BBB                    |
       1 |AAAAAAAAAAAAAAAAAAAABBBAAAAAAAAAAAAAAAAAAAA +-|
     0.8 |-+               BBBB                       +-|
         |               BBB                            |
     0.6 |-+          BBBB                            +-|
     0.4 |-+    BBBBBBB                               +-|
         |  BBBB                                        |
     0.2 |-B  +    +     +    +    +    +     +    +  +-|
       0 +----------------------------------------------+
         0    10   20    30   40   50   60    70   80   90
                         Workload Number
```

The last plot shows the "latest" snapshot for all benchmarks ("Workload" on the
plot), sorting them by speedup. That is, it shows the speedup of both Inductor
and Pytorch/XLA over the oldest Inductor data point that we have in the JSONL
files. (Note: to reiterate, because we are plotting data from single day,
Inductor gets speedup == 1 for all benchmarks). This plot also shows the
correctness gap between Pytorch/XLA and Inductor; there are benchmarks that do
run on Inductor but not on Pytorch/XLA.

## Continuous Integration Tests

Benchmark-related tests run by CI are located at `xla/test/benchmarks`.
To run the tests locally, do `$ make -C xla/test/benchmarks`.
