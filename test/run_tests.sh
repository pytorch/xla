#!/bin/bash
set -exo pipefail
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
MAX_GRAPH_SIZE=500
GRAPH_CHECK_FREQUENCY=100
VERBOSITY=2

while getopts 'LM:C:V:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
      ;;
    M)
      MAX_GRAPH_SIZE=$OPTARG
      ;;
    C)
      GRAPH_CHECK_FREQUENCY=$OPTARG
      ;;
    V)
      VERBOSITY=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))

export TRIM_GRAPH_SIZE=$MAX_GRAPH_SIZE
export TRIM_GRAPH_CHECK_FREQUENCY=$GRAPH_CHECK_FREQUENCY
export XLA_TEST_DIR=$CDIR
export PYTORCH_TEST_WITH_SLOW=1

function run_opbyop {
  echo "Running in OpByOp mode ..."
  XLA_GET_TENSORS_OPBYOP=1 XLA_SYNC_TENSORS_OPBYOP=1 "$@"
}

function run_dynamic {
  XLA_EXPERIMENTAL="nonzero:masked_select" "$@"
}

function run_all_tests {
  python3 "$CDIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  run_dynamic python3 "$CDIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  python3 "$CDIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  python3 "$CDIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  python3 "$CDIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  python3 "$CDIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
  run_dynamic python3 "$CDIR/../../test/test_type_promotion.py" "$@" -v TestTypePromotionXLA
  run_dynamic python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_opbyop python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  python3 "$CDIR/test_mp_replication.py" "$@"
}

if [ "$LOGFILE" != "" ]; then
  run_all_tests 2>&1 | tee $LOGFILE
else
  run_all_tests
fi
