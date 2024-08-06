#!/bin/bash
# Pytorch/XLA Nightly Benchmark Runner.

set -ex

ACCELERATOR=a100
OUTPUT_DIR=${HOME:?}
WORKSPACE=$(date --utc +%Y-%m-%d)
REPEAT=8
ENABLE_PROFILING=

while getopts 'A:O:PR:T:W:' OPTION
do
  case ${OPTION?} in
    A)
      ACCELERATOR=${OPTARG:?}
      ;;
    O)
      OUTPUT_DIR=${OPTARG:?}
      ;;
    P)
      ENABLE_PROFILING=1
      ;;
    R)
      REPEAT=${OPTARG:?}
      ;;
    T)
      # Avoid printing the token; re-enable printing later.
      { set +x; } 2>/dev/null
      export HUGGING_FACE_HUB_TOKEN=${OPTARG:?}
      set -x
      ;;
    W)
      WORKSPACE=${OPTARG:?}
      ;;
  esac
done

NIGHTLY_RUNS=nightly_runs
NIGHTLY_RESULTS=nightly_results
if [[ ${ENABLE_PROFILING?} ]]; then
  NIGHTLY_RUNS=nightly_profiling_runs
  NIGHTLY_RESULTS=nightly_profiling_results
fi
WORKSPACE_DIR=${OUTPUT_DIR:?}/${NIGHTLY_RUNS:?}/${WORKSPACE:?}
BM_DIR=${WORKSPACE_DIR:?}/pytorch/xla/benchmarks

# Intermediate results, which are processed to generate reports.
WORKSPACE_RESULTS_DIR=${WORKSPACE_DIR:?}/experiment_results

# Final data files and reports go here.
NIGHTLY_RESULTS_DIR=${OUTPUT_DIR:?}/${NIGHTLY_RESULTS:?}

# Init workspace
#
# Sometimes a run fails halfway. Typically this is because
# experiment_runner crashes. We then fix the problem and
# run the script again, which skips the build phase.
IS_FRESH_RUN=1  # Set to null below; read with ${IS_FRESH_RUN?}.
if [ -d ${WORKSPACE_DIR:?} ]; then
  IS_FRESH_RUN=
fi

if [[ ${IS_FRESH_RUN?} ]]; then
  rm -rf ${HOME:?}/.cache/bazel
fi

mkdir -p ${WORKSPACE_DIR:?}
cd ${WORKSPACE_DIR:?}

ENV_DIR=env
if [[ ${IS_FRESH_RUN?} ]]; then
  python3 -m venv ${ENV_DIR:?}
fi
source ${ENV_DIR:?}/bin/activate

# Download and build everything
if [[ ${IS_FRESH_RUN?} ]]; then
  # Install deps
  pip install --upgrade pip

  TIMESTAMP=$(date +%s)
  # Clone repos first so that their HEAD is as close as possible to $TIMESTAMP.
  git clone https://github.com/pytorch/pytorch.git
  git clone https://github.com/pytorch/xla.git pytorch/xla
  git clone https://github.com/pytorch/vision.git
  git clone https://github.com/pytorch/audio.git
  git clone https://github.com/pytorch/benchmark.git

  # Set up pytorch
  cd pytorch
  pip install -r requirements.txt
  make triton
  USE_CUDA=1 python setup.py develop
  cd ..

  # Set up pytorch/xla
  cd pytorch/xla
  # Query local compute capability. If that fails, assign a sane default.
  LOCAL_CAP=compute_$(nvidia-smi --query-gpu=compute_cap --format=csv | \
    tail -1 | sed 's/\.//g' | grep -E '^[0-9]{2}$' || echo '80')
  XLA_CUDA=1 TF_CUDA_COMPUTE_CAPABILITIES=${LOCAL_CAP:?} python setup.py develop
  cd ../..

  # Set up torchbench deps.
  cd vision
  python setup.py develop
  cd ..
  cd audio
  python setup.py develop
  cd ..

  # Set up torchbench
  cd benchmark
  USE_CUDA=1 python install.py
  cd ..

  # Apply local patches
  cd benchmark
  git apply ../pytorch/xla/benchmarks/patches/mismatched_batch_size.patch
  cd ..
else
  # Grab the timestamp from the first result, if it exists.
  # Otherwise take the current timestamp.
  TIMESTAMP=$(head -1 ${WORKSPACE_RESULTS_DIR:?}/results.jsonl | \
    sed -E 's|.*\"timestamp\": ([0-9.]+).*|\1|' | \
    grep -E '^[0-9.]+$' || date +%s)
fi

# Stabilize clock freqs
sudo nvidia-smi --lock-gpu-clocks=1200,1200

# Note: this doesn't work on GCP because it's a VM.
# Moreover, we should look into disabling turbo boost if possible.
#   sudo cpupower frequency-set --governor performance

PROFILING_FLAGS=
if [[ ${ENABLE_PROFILING?} ]]; then
  PROFILING_FLAGS="--dump-dynamo-counters \
    --collect-dynamo-counters \
    --dump-pytorch-profiles \
    --dump-pytorch-xla-metrics \
    --profile-cuda-cpu \
    --profile-cuda-cpu-individual-ops"
fi

# Run the experiments
cd pytorch
# Note: to avoid running in Eager mode (i.e. --xla=None --dynamo=None),
# we split experiment_runner.py's invocation in two.
#
# Inference + Training: XLA Lazy tensors, XLA+XLA_Eval Dynamo.
python xla/benchmarks/experiment_runner.py \
       --test=eval --test=train \
       --xla=PJRT \
       --dynamo=None --dynamo=openxla \
       --suite-name=torchbench --accelerator=cuda \
       --output-dirname=${WORKSPACE_RESULTS_DIR:?} \
       --repeat=${REPEAT:?} --print-subprocess \
       --timestamp=${TIMESTAMP:?} ${PROFILING_FLAGS?}
# Inference + Training: Inductor Dynamo.
python xla/benchmarks/experiment_runner.py \
       --test=eval --test=train \
       --xla=None \
       --dynamo=inductor \
       --suite-name=torchbench --accelerator=cuda \
       --output-dirname=${WORKSPACE_RESULTS_DIR:?} \
       --repeat=${REPEAT:?} --print-subprocess \
       --timestamp=${TIMESTAMP:?} ${PROFILING_FLAGS?}
cd ..

# Run Llama2 benchmarks.
python ${BM_DIR:?}/llama.py --workspace_dir=${WORKSPACE_DIR:?}

# Gather results and generate reports
REPORTS_DIR=${NIGHTLY_RESULTS_DIR:?}/reports/${WORKSPACE:?}
mkdir -p ${REPORTS_DIR:?}
cp ${WORKSPACE_RESULTS_DIR:?}/results.jsonl \
   ${NIGHTLY_RESULTS_DIR:?}/${WORKSPACE:?}.jsonl

PYTORCH_GIT_REV=$(git -C pytorch rev-parse --short HEAD)
XLA_GIT_TAG=$(git -C pytorch/xla describe --tags --always)
GIT_TAGS="PT: ${PYTORCH_GIT_REV:?} XLA: ${XLA_GIT_TAG:?}"

COMMON_TITLE_PREFIX=
if [[ ${ENABLE_PROFILING?} ]]; then
  COMMON_TITLE_PREFIX="[Profiling ON] "
fi

INFERENCE_BACKENDS_CMD='--backends inductor openxla+dynamo openxla+lazytensor'
TRAINING_BACKENDS_CMD='--backends inductor openxla+dynamo openxla+lazytensor'

# Skip result files coming from one-off runs.
INPUT_JSONL_FILES=$(ls ${NIGHTLY_RESULTS_DIR:?}/*.jsonl | \
  grep '[0-9]\+-[0-9]\+-[0-9]\+\.jsonl')

for testname in inference training; do
  for report in latest histogram speedup; do
    for format in csv svg; do
      for tier in '' 1; do
        TITLE_PREFIX=
        TIER_CMD=
        TIER_FILE_SUFFIX=
        if [[ ${tier?} ]]; then
          TITLE_PREFIX="${COMMON_TITLE_PREFIX?}Tier${tier?} "
          TIER_CMD=--filter-by-tier=${tier:?}
          TIER_FILE_SUFFIX=-tier${tier:?}
        fi

        TITLE="(${testname:?})"
        WIDTH=9
        HEIGHT=7
        if [ "${report:?}" == "latest" ]; then
          TITLE="${WORKSPACE:?} (${testname:?}) ${GIT_TAGS:?}"
          if [[ -z ${tier?} ]]; then
            WIDTH=15
            HEIGHT=8
          fi
        fi
        BACKENDS_CMD=
        if [ "${testname:?}" = 'inference' ]; then
            BACKENDS_CMD="${INFERENCE_BACKENDS_CMD:?}"
        else
            BACKENDS_CMD="${TRAINING_BACKENDS_CMD:?}"
        fi
        python ${BM_DIR:?}/aggregate.py --accelerator=${ACCELERATOR:?} \
               --report=${report:?} --test=${testname:?} --format=${format:?} \
               --title="${TITLE_PREFIX?}${TITLE:?}" \
               --fig-height=${HEIGHT:?} --fig-width=${WIDTH:?} \
               ${TIER_CMD?} \
               ${BACKENDS_CMD:?} -- \
               ${INPUT_JSONL_FILES:?} \
               > ${REPORTS_DIR:?}/${ACCELERATOR:?}-${testname:?}-${report:?}${TIER_FILE_SUFFIX?}.${format:?}
      done
    done
  done
done

# Generate Llama2 output.
for testname in inference; do
  for report in latest_grouped; do
    for format in csv svg tab; do
      BACKENDS_CMD=
      if [ "${testname:?}" = 'inference' ]; then
        BACKENDS_CMD="${INFERENCE_BACKENDS_CMD:?}"
      else
        BACKENDS_CMD="${TRAINING_BACKENDS_CMD:?}"
      fi
      python ${BM_DIR:?}/aggregate.py --accelerator=${ACCELERATOR:?} \
             --report=${report:?} --test=${testname:?} --format=${format:?} \
             --title="${COMMON_TITLE_PREFIX?}Llama2 (${testname:?})" \
             --filter='^llama2\.' \
             ${BACKENDS_CMD:?} -- \
             ${INPUT_JSONL_FILES:?} \
             > ${REPORTS_DIR:?}/${ACCELERATOR:?}-${testname:?}-${report:?}-llama2.${format:?}
    done
  done
done
