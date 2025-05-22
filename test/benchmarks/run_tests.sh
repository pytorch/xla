#!/bin/bash
set -ex
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_benchmarks_test.log
VERBOSITY=0

BENCHMARKS_DIR="$CDIR/../../benchmarks/"
TORCH_XLA_DIR=$(cd ~; dirname "$(python -c 'import torch_xla; print(torch_xla.__file__)')")

# Make benchmark module available as it is not part of torch_xla.
export PYTHONPATH=$PYTHONPATH:$BENCHMARKS_DIR

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `1` to make the CI tests continue on error.
# This will allow you to see all the failures on your PR, not stopping with the first
# test failure like the default behavior.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  set +e
fi

while getopts 'LV:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
    ;;
    V)
      VERBOSITY=$OPTARG
    ;;
  esac
done
shift $(($OPTIND - 1))

function run_coverage {
  if [ "${USE_COVERAGE:-0}" != "0" ]; then
    coverage run --source="$TORCH_XLA_DIR,$BENCHMARKS_DIR" -p "$@"
  else
    python3 "$@"
  fi
}

function run_make_tests {
  MAKE_V=""
  if [ "$VERBOSITY" != "0" ]; then
    MAKE_V="V=$VERBOSITY"
  fi
  make -C $CDIR $MAKE_V all
}

function run_python_tests {
  # HACK: don't confuse local `torch_xla` folder with installed package
  # Python 3.11 has the permanent fix: https://stackoverflow.com/a/73636559
  pushd $CDIR
  run_coverage "test_experiment_runner.py"
  run_coverage "test_benchmark_experiment.py"
  run_coverage "test_benchmark_model.py"
  run_coverage "test_result_analyzer.py"
  popd
}

function run_tests {
  run_make_tests
  run_python_tests
}

if [ "$LOGFILE" != "" ]; then
  run_tests 2>&1 | tee $LOGFILE
else
  run_tests
fi
