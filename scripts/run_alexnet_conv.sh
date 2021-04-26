#!/bin/bash
./scripts/run_conv_experiment.sh 0 alexnet seg all 3 &
./scripts/run_conv_experiment.sh 1 alexnet seg front 3 &
./scripts/run_conv_experiment.sh 2 alexnet seg back 3 &