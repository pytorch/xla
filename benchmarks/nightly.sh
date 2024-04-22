#!/bin/bash
# Pytorch/XLA Nightly Benchmark Runner.

set -ex

ACCELERATOR=a100
OUTPUT_DIR=${HOME:?}
WORKSPACE=$(date --utc +%Y-%m-%d)
REPEAT=8

while getopts 'A:O:R:W:' OPTION
do
  case ${OPTION?} in
    A)
      ACCELERATOR=${OPTARG:?}
      ;;
    O)
      OUTPUT_DIR=${OPTARG:?}
      ;;
    R)
      REPEAT=${OPTARG:?}
      ;;
    W)
      WORKSPACE=${OPTARG:?}
      ;;
  esac
done

WORKSPACE_DIR=${OUTPUT_DIR:?}/nightly_runs/${WORKSPACE:?}

# Intermediate results, which are processed to generate reports.
WORKSPACE_RESULTS_DIR=${WORKSPACE_DIR:?}/experiment_results

# Final data files and reports go here.
NIGHTLY_RESULTS_DIR=${OUTPUT_DIR:?}/nightly_results

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
  XLA_CUDA=1 python setup.py develop
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
		sed -E 's|.*\"timestamp\": ([0-9.]+).*|\1|' || date +%s)
fi

# Stabilize clock freqs
sudo nvidia-smi --lock-gpu-clocks=1200,1200

# Note: this doesn't work on GCP because it's a VM.
# Moreover, we should look into disabling turbo boost if possible.
#   sudo cpupower frequency-set --governor performance

# Run the experiments
cd pytorch
python xla/benchmarks/experiment_runner.py \
       --dynamo=openxla --dynamo=openxla_eval --dynamo=inductor \
       --xla=PJRT --xla=None --test=eval --test=train \
       --suite-name=torchbench --accelerator=cuda \
       --output-dirname=${WORKSPACE_RESULTS_DIR:?} \
       --repeat=${REPEAT:?} --print-subprocess \
       --timestamp=${TIMESTAMP:?}
cd ..

# Gather results and generate reports
REPORTS_DIR=${NIGHTLY_RESULTS_DIR:?}/reports/${WORKSPACE:?}
mkdir -p ${REPORTS_DIR:?}
cp ${WORKSPACE_RESULTS_DIR:?}/results.jsonl \
   ${NIGHTLY_RESULTS_DIR:?}/${WORKSPACE:?}.jsonl

PYTORCH_GIT_REV=$(git -C pytorch rev-parse --short HEAD)
XLA_GIT_TAG=$(git -C pytorch/xla describe --tags)
GIT_TAGS="PT: ${PYTORCH_GIT_REV:?} XLA: ${XLA_GIT_TAG:?}"
BM_DIR=${WORKSPACE_DIR:?}/pytorch/xla/benchmarks

for testname in inference training; do
  for report in latest histogram speedup; do
    for format in csv svg; do
      for tier in '' 1; do
	# Note: these (as well as $tier) can be null; read them with ${FOO?}
	# instead of ${FOO:?}.
	TITLE_PFX=
	TIER_CMD=
	TIER_FILE_SUFFIX=
	if [[ ${tier?} ]]; then
	  TITLE_PFX="Tier${tier?} "
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
	python ${BM_DIR:?}/aggregate.py --accelerator=${ACCELERATOR:?} \
               --report=${report:?} --test=${testname:?} --format=${format:?} \
	       --title="${TITLE_PFX?}${TITLE:?}" \
	       --fig-height=${HEIGHT:?} --fig-width=${WIDTH:?} \
	       ${TIER_CMD?} \
	       ${NIGHTLY_RESULTS_DIR:?}/*.jsonl \
               > ${REPORTS_DIR:?}/${ACCELERATOR:?}-${testname:?}-${report:?}${TIER_FILE_SUFFIX?}.${format:?}
      done
    done
  done
done
