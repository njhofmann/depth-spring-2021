#!/bin/bash
./scripts/run_conv_experiment.sh 0 resnet seg all 3 &
./scripts/run_conv_experiment.sh 1 resnet seg front 3 &
./scripts/run_conv_experiment.sh 2 resnet seg back 3 &