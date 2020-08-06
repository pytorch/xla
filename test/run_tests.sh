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
export TORCH_TEST_DEVICES="$CDIR/pytorch_test_base.py"
export PYTORCH_TEST_WITH_SLOW=1
export XLA_DUMP_FATAL_STACK=1

function run_test {
  "$@"
}

function run_opbyop {
  echo "Running in OpByOp mode: $@"
  XLA_GET_TENSORS_OPBYOP=1 XLA_SYNC_TENSORS_OPBYOP=1 run_test "$@"
}

function run_dynamic {
  echo "Running in DynamicShape mode: $@"
  XLA_EXPERIMENTAL="nonzero:masked_select" run_test "$@"
}

function run_all_tests {
  run_dynamic python3 "$CDIR/../../test/test_torch.py" "$@" -v TestViewOpsXLA
  run_test python3 "$CDIR/../../test/test_torch.py" "$@" -v TestTorchDeviceTypeXLA
  run_dynamic python3 "$CDIR/../../test/test_torch.py" "$@" -v TestDevicePrecisionXLA
  run_test python3 "$CDIR/../../test/test_torch.py" "$@" -v TestTensorDeviceOpsXLA
  run_test python3 "$CDIR/../../test/test_indexing.py" "$@" -v TestIndexingXLA
  run_test python3 "$CDIR/../../test/test_indexing.py" "$@" -v NumpyTestsXLA
  run_dynamic python3 "$CDIR/../../test/test_nn.py" "$@" -v TestNNDeviceTypeXLA
  run_dynamic python3 "$CDIR/../../test/test_type_promotion.py" "$@" -v TestTypePromotionXLA
  run_dynamic python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_opbyop python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  run_test python3 "$CDIR/test_mp_replication.py"
  run_test python3 "$CDIR/test_mp_all_to_all.py"
  run_test python3 "$CDIR/test_mp_collective_permute.py"
  run_test python3 "$CDIR/test_mp_all_gather.py"
  run_test python3 "$CDIR/test_mp_distributed_mm.py"
  run_test python3 "$CDIR/test_mp_rendezvous.py"
  run_test python3 "$CDIR/test_mp_save.py"
  run_test python3 "$CDIR/test_mp_mesh_reduce.py"
}

if [ "$LOGFILE" != "" ]; then
  run_all_tests 2>&1 | tee $LOGFILE
else
  run_all_tests
fi
