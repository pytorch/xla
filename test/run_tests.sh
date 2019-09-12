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

if [ "$LOGFILE" != "" ]; then
  python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY 2>&1 | tee $LOGFILE
  python3 "$CDIR/test_mp_replication.py" "$@" 2>&1 | tee $LOGFILE
else
  python3 "$CDIR/test_operations.py" "$@" --verbosity=$VERBOSITY
  python3 "$CDIR/test_mp_replication.py" "$@"
fi
