#!/usr/bin/env bash

set -ex

DATE=$(date +"%Y_%m_%d_%H_%M")

OUT_PATH=xla/benchmarks/bm_results/single_graph/$DATE
mkdir -p $OUT_PATH

python new_xla/benchmarks/experiment_runner.py \
    --dynamo=inductor --dynamo=openxla \
    --xla=None --xla=PJRT \
    --test=eval \
    --filter-by-single-graph \
    --pure-wall-time \
    --suite-name=torchbench \
    --accelerator=cuda \
    --output-dirname=$OUT_PATH \
    --repeat=5 \
    --print-subprocess \
    --no-resume \
    > $OUT_PATH/stdout.txt 2> $OUT_PATH/stderr.txt

python3 xla/benchmarks/result_analyzer.py \
    --output-dirname=$OUT_PATH \
    --database=$OUT_PATH/$DATE.csv
