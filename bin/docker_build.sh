#!/usr/bin/env bash

DIR=$( dirname -- "$0"; )
pushd $DIR
docker build . -t pd/seminar
popd
