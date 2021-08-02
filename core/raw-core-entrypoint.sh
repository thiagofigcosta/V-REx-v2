#!/bin/bash

ulimit -s unlimited
export OMP_STACKSIZE=4G

program_args=""

debug=0
profiling=0

while [ $# -gt 0 ] ; do
  case $1 in
    --debug) 
        debug=1
        shift 1 # shift once, for argu,ment
    ;;

    --profiling) 
        profiling=1
        shift 1 # shift once, for argu,ment
    ;;

    *)
        program_args="$program_args $1"
        shift 1
    ;;
  esac
done


if [ $debug -ne 0 ] ; then 
    gdb --args /vrex/src/bin/vrex $program_args
elif  [ $profiling -ne 0 ] ; then 
    valgrind --leak-check=full --track-origins=yes /vrex/src/bin/vrex $program_args
else
    ./vrex/src/bin/vrex $program_args
fi