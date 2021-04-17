#!/bin/bash

# resnet
python -m src.main -m resnet -c rgbd -s -bs 4 -e 100 -mn resnet_rgbd_seg_base_1 > outputs/resnet_rgbd_seg_base_1.txt 2>&1
python -m src.main -m resnet -c rgbd -s -bs 4 -e 100 -mn resnet_rgbd_seg_base_2 > outputs/resnet_rgbd_seg_base_2.txt 2>&1
python -m src.main -m resnet -c rgbd -s -bs 4 -e 100 -mn resnet_rgbd_seg_base_3 > outputs/resnet_rgbd_seg_base_3.txt 2>&1

python -m src.main -m resnet -c rgb -s -bs 4 -e 100 -mn resnet_rgb_seg_base_1 > outputs/resnet_rgb_seg_base_1.txt 2>&1
python -m src.main -m resnet -c rgb -s -bs 4 -e 100 -mn resnet_rgb_seg_base_2 > outputs/resnet_rgb_seg_base_2.txt 2>&1
python -m src.main -m resnet -c rgb -s -bs 4 -e 100 -mn resnet_rgb_seg_base_3 > outputs/resnet_rgb_seg_base_3.txt 2>&1

python -m src.main -m resnet -c d -s -bs 4 -e 100 -mn resnet_d_seg_base_1 > outputs/resnet_d_seg_base_1.txt 2>&1
python -m src.main -m resnet -c d -s -bs 4 -e 100 -mn resnet_d_seg_base_2 > outputs/resnet_d_seg_base_2.txt 2>&1
python -m src.main -m resnet -c d -s -bs 4 -e 100 -mn resnet_d_seg_base_3 > outputs/resnet_d_seg_base_3.txt 2>&1

# vgg
python -m src.main -m vgg -c rgbd -s -bs 4 -e 100 -mn vgg_rgbd_seg_base_1 > outputs/vgg_rgbd_seg_base_1.txt 2>&1
python -m src.main -m vgg -c rgbd -s -bs 4 -e 100 -mn vgg_rgbd_seg_base_2 > outputs/vgg_rgbd_seg_base_2.txt 2>&1
python -m src.main -m vgg -c rgbd -s -bs 4 -e 100 -mn vgg_rgbd_seg_base_3 > outputs/vgg_rgbd_seg_base_3.txt 2>&1

python -m src.main -m vgg -c rgb -s -bs 4 -e 100 -mn vgg_rgb_seg_base_1 > outputs/vgg_rgb_seg_base_1.txt 2>&1
python -m src.main -m vgg -c rgb -s -bs 4 -e 100 -mn vgg_rgb_seg_base_2 > outputs/vgg_rgb_seg_base_2.txt 2>&1
python -m src.main -m vgg -c rgb -s -bs 4 -e 100 -mn vgg_rgb_seg_base_3 > outputs/vgg_rgb_seg_base_3.txt 2>&1

python -m src.main -m vgg -c d -s -bs 4 -e 100 -mn vgg_d_seg_base_1 > outputs/vgg_d_seg_base_1.txt 2>&1
python -m src.main -m vgg -c d -s -bs 4 -e 100 -mn vgg_d_seg_base_2 > outputs/vgg_d_seg_base_2.txt 2>&1
python -m src.main -m vgg -c d -s -bs 4 -e 100 -mn vgg_d_seg_base_3 > outputs/vgg_d_seg_base_3.txt 2>&1

# densenet
python -m src.main -m dense -c rgbd -s -bs 4 -e 100 -mn dense_rgbd_seg_base_1 > outputs/dense_rgbd_seg_base_1.txt 2>&1
python -m src.main -m dense -c rgbd -s -bs 4 -e 100 -mn dense_rgbd_seg_base_2 > outputs/dense_rgbd_seg_base_2.txt 2>&1
python -m src.main -m dense -c rgbd -s -bs 4 -e 100 -mn dense_rgbd_seg_base_3 > outputs/dense_rgbd_seg_base_3.txt 2>&1

python -m src.main -m dense -c rgb -s -bs 4 -e 100 -mn dense_rgb_seg_base_1 > outputs/dense_rgb_seg_base_1.txt 2>&1
python -m src.main -m dense -c rgb -s -bs 4 -e 100 -mn dense_rgb_seg_base_2 > outputs/dense_rgb_seg_base_2.txt 2>&1
python -m src.main -m dense -c rgb -s -bs 4 -e 100 -mn dense_rgb_seg_base_3 > outputs/dense_rgb_seg_base_3.txt 2>&1

python -m src.main -m dense -c d -s -bs 4 -e 100 -mn dense_d_seg_base_1 > outputs/dense_d_seg_base_1.txt 2>&1
python -m src.main -m dense -c d -s -bs 4 -e 100 -mn dense_d_seg_base_2 > outputs/dense_d_seg_base_2.txt 2>&1
python -m src.main -m dense -c d -s -bs 4 -e 100 -mn dense_d_seg_base_3 > outputs/dense_d_seg_base_3.txt 2>&1