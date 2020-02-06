#!/bin/bash

# Usage: ./change_tpu_runtime.sh <project_name> <tpu_zone> <tpu_name> <version_date>
# Use with 'pytorch-nightly' TPU version only

set -e  # Fail on first failure
set -x  # echo commands

PROJECT=$1
ZONE=$2
TPU_NAME=$3
DATE=$4  # YYYYMMDD

# Change all workers runtime and store PIDs to wait on
GVM_IPS=$(gcloud --project="${PROJECT:?}" compute tpus describe "${TPU_NAME:?}" --zone="${ZONE:?}" --format='value(networkEndpoints[].ipAddress)')
for ip in $(echo "${GVM_IPS:?}" | tr ';' '\n')
do
  curl -X POST "http://${ip:?}:8475/requestversion/TPU-dev${DATE:?}" &
  pids[${i}]=$!
done

# Wait for all runtimes to be updated
for pid in ${pids[*]}; do
  wait $pid
done

