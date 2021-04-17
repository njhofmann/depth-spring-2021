#!/bin/bash

# resnet
python -m src.main -m resnet -c rgbd -b -bs 4 -dco all -mn resnet_rgbd_seg_base_1 > outputs/resnet_rgbd_seg_base_1.txt 2>&1
python -m src.main -m resnet -c rgbd -b -bs 4 -dco all -mn resnet_rgbd_seg_base_2 > outputs/resnet_rgbd_seg_base_2.txt 2>&1
python -m src.main -m resnet -c rgbd -b -bs 4 -dco all -mn resnet_rgbd_seg_base_3 > outputs/resnet_rgbd_seg_base_3.txt 2>&1

# vgg
python -m src.main -m vgg -c rgbd -b -bs 4 -dco all -mn vgg_rgbd_seg_base_1 > outputs/vgg_rgbd_seg_base_1.txt 2>&1
python -m src.main -m vgg -c rgbd -b -bs 4 -dco all -mn vgg_rgbd_seg_base_2 > outputs/vgg_rgbd_seg_base_2.txt 2>&1
python -m src.main -m vgg -c rgbd -b -bs 4 -dco all -mn vgg_rgbd_seg_base_3 > outputs/vgg_rgbd_seg_base_3.txt 2>&1

# densenet
python -m src.main -m dense -c rgbd -b -bs 4 -dco all -mn dense_rgbd_seg_base_1 > outputs/dense_rgbd_seg_base_1.txt 2>&1
python -m src.main -m dense -c rgbd -b -bs 4 -dco all -mn dense_rgbd_seg_base_2 > outputs/dense_rgbd_seg_base_2.txt 2>&1
python -m src.main -m dense -c rgbd -b -bs 4 -dco all -mn dense_rgbd_seg_base_3 > outputs/dense_rgbd_seg_base_3.txt 2>&1