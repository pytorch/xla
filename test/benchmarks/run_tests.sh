#!/bin/bash
set -ex
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_benchmarks_test.log
VERBOSITY=0

# Make benchmark module available as it is not part of torch_xla.
export PYTHONPATH=$PYTHONPATH:$CDIR/../../benchmarks/

# Note [Keep Going]
#
# Set the `CONTINUE_ON_ERROR` flag to `true` to make the CircleCI tests continue on error.
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

function run_make_tests {
  MAKE_V=""
  if [ "$VERBOSITY" != "0" ]; then
    MAKE_V="V=$VERBOSITY"
  fi
  make -C $CDIR $MAKE_V all
}

function run_python_tests {
  python3 "$CDIR/test_experiment_runner.py"
  python3 "$CDIR/test_benchmark_experiment.py"
  python3 "$CDIR/test_benchmark_model.py"
  python3 "$CDIR/test_result_analyzer.py"
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
