#!/usr/bin/env bash

set -ex

OUT_PATH=xla/benchmark/bm_results/$(date +"%Y_%m_%d_%H_%M")
mkdir -p $OUT_PATH

python xla/benchmark/experiment_runner.py \
    --dynamo=inductor --dynamo=openxla_eval --dynamo=openxla --dynamo=None \
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

python3 xla/benchmark/result_analyzer.py \
    --output-dirname=$OUT_PATH \
    --database=$OUT_PATH/database.csv
