#!/bin/bash
./scripts/run_conv_experiment.sh 0 vgg seg all 3 &
./scripts/run_conv_experiment.sh 1 vgg seg front 3 &
./scripts/run_conv_experiment.sh 2 vgg seg back 3 &