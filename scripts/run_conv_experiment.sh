#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

model=$2
mode=$3
conv=$4

for i in `eval echo {0..$5}`;
do
  save_name=${model}_rgbd_${mode}_${conv}_${i}
  echo running $save_name on gpu $1
  python -m src.main -m $model -c rgbd --$mode -e 100 -bs 4 -dco $conv -mn $save_name > outputs/${save_name}.txt
done

