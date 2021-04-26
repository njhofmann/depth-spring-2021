#!/bin/bash
./scripts/run_base_experiment.sh 0 vgg seg rgbd 3 &
./scripts/run_base_experiment.sh 1 vgg seg rgb 3 &
./scripts/run_base_experiment.sh 2 vgg seg d 3 &