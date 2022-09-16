#!/usr/bin/env bash

docker run -d --name pd-seminar --volume `pwd`:/home pd/seminar
