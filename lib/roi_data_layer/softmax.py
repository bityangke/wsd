# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

#bottom[0]=(128,21,1,1) #classification scores
#bottom[1]=(21) #image labels
#for the moment I assume an image for each batch, but not a good assumption!!! 

import caffe
import numpy as np

def softmax(x,axis=-1):
    #e_x = np.exp(x - np.max(x,axis=axis))
    #out = e_x / e_x.sum(axis=axis)
    mmax = np.max(x,axis=axis,keepdims=True)
    #print mmax.shape,(x-mmax).shape,(np.log(np.sum(np.exp(x - mmax),axis=axis))).shape
    out = mmax.squeeze() + np.log(np.sum(np.exp(x - mmax),axis=axis))
    return out

def weights(x,axis=-1):
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    #print e_x
    out = e_x / e_x.sum(axis=axis,keepdims=True)
    return out

class MySoftMaxLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(1, bottom[0].channels,
                bottom[0].height, bottom[0].width)
        # check input dimensions match
        #if bottom[0].data.shape[1] != bottom[1].data.shape[0]:
        #    raise Exception("There should be a 0-1 label per class, per image.")
        # difference is shape of inputs
        #self.diff = np.zeros(bottom[0].data.shape(1), dtype=np.float32)
        # loss output is scalar
        #top[0].reshape(bottom[0].data.shape(1))

    def forward(self, bottom, top):
        #self.diff[...] = bottom[0].data - bottom[1].data
        #top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
      
        #softmax in dimension 1 
        max_cls = softmax(bottom[0].data,axis=0)
        top[0].data[0,:,0,0] = max_cls
        self.weights = weights(bottom[0].data,axis=0)
        #cross entropy in dimension 0
        

    def backward(self, top, propagate_down, bottom):
        if propagate_down:
          bottom[0].diff[...] = top[0].diff.squeeze()*self.weights

