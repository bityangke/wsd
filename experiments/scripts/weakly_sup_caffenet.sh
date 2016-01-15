#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="experiments/logs/weakly_sup_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu $1 \
  --solver models/CaffeNet/weakly_supervised/solver.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/weakly_sup6.yml

time ./tools/test_net.py --gpu $1 \
  --def models/CaffeNet/no_bbox_reg/test.prototxt \
  --net output/weakly_supervised6/voc_2007_trainval/caffenet_fast_rcnn_no_bbox_reg_iter_40000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/weakly_sup6.yml
