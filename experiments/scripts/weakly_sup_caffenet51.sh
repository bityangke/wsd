#!/bin/bash

. /home/lear/mpederso/.bashrc;
setpath

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU=`echo gpu_getIDs.sh`
LOG="./experiments/logs/weakly_sup_caffenet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu `$GPU` \
  --solver models/CaffeNet/weakly_supervised/solverHakanBeta2.prototxt \
  --weights data/imagenet_models/CaffeNet.v2.caffemodel \
  --imdb voc_2007_trainval \
  --cfg experiments/cfgs/weakly_sup51.yml \
  --iters 100000

time ./tools/test_net.py --gpu `$GPU` \
  --def models/CaffeNet/weakly_supervised/testHakanBeta.prototxt \
  --net output/weakly_supervised51/voc_2007_trainval/caffenet_fast_rcnn_no_bbox_reg_iter_100000.caffemodel \
  --imdb voc_2007_test \
  --cfg experiments/cfgs/weakly_sup51.yml
