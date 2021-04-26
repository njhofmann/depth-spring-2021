#!/bin/bash
./scripts/run_base_experiment.sh 3 alexnet seg rgbd 3 &
./scripts/run_base_experiment.sh 1 alexnet seg rgb 3 &
./scripts/run_base_experiment.sh 2 alexnet seg d 3 &