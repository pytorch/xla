#!/usr/bin/env bash

set -ex

DATE=$(date +"%Y_%m_%d_%H_%M")

OUT_PATH=xla/benchmarks/bm_results/$DATE
mkdir -p $OUT_PATH

python xla/benchmarks/experiment_runner.py \
    --dynamo=inductor --dynamo=openxla \
    --xla=None --xla=PJRT \
    --test=eval --test=train \
    --filter-by-tier=1 --filter-by-tier=2 --filter-by-tier=3 \
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
