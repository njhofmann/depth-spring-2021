#!/bin/bash
./scripts/run_base_experiment.sh 0 resnet seg rgbd 3 &
./scripts/run_base_experiment.sh 1 resnet seg rgb 3 &
./scripts/run_base_experiment.sh 2 resnet seg d 3 &