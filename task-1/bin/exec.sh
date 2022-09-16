#!/usr/bin/env bash
DIR=$(dirname -- "$0"; )
mpiexec --allow-run-as-root -n $1 $DIR/MpiIntegrate -T $2 -N $3