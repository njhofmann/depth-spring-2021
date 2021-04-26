#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

model=$2
mode=$3
channels=$4

for i in `eval echo {1..$5}`;
do
  save_name=${model}_${channels}_${mode}_${i}
  echo running $save_name on gpu $1
  python -m src.main -m $model -c $channels --$mode -e 100 -bs 4 -mn $save_name > outputs/${save_name}.txt
done

