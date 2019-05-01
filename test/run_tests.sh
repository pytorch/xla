#!/bin/bash
set -ex
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
MAX_GRAPH_SIZE=1000

while getopts 'LM:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
      ;;
    M)
      MAX_GRAPH_SIZE=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))

export TRIM_GRAPH_SIZE=$MAX_GRAPH_SIZE

if [ "$LOGFILE" != "" ]; then
  python "$CDIR/test_operations.py" "$@" 2>&1 | tee $LOGFILE
else
  python "$CDIR/test_operations.py" "$@"
fi
