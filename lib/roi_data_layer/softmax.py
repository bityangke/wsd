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

import yaml
import caffe
import numpy as np
from fast_rcnn.config import cfg

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

def betaweights(x,beta,axis=-1):
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))**beta
    #print e_x
    ssum=e_x.sum(axis=axis,keepdims=True)
    out = e_x / ssum
    entr = np.sum(e_x*x,axis=axis,keepdims=True)
    return out,ssum,entr

#def betaweights2(x,beta,axis=-1):
#    e_x = (x - np.max(x,axis=axis,keepdims=True))**beta
#    #print e_x
#    ssum=e_x.sum(axis=axis,keepdims=True)
#    out = e_x / (ssum+0.000001)
#    #entr = np.sum(e_x*x,axis=axis,keepdims=True)
#    return out

def betaweights2(x,beta,axis=-1):
    e_x = beta**(x - np.max(x,axis=axis,keepdims=True))
    #print e_x
    ssum=e_x.sum(axis=axis,keepdims=True)
    out = e_x / ssum
    #entr = np.sum(e_x*x,axis=axis,keepdims=True)
    return out
    
class MySoftMaxLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(1, bottom[0].channels,
                bottom[0].height, bottom[0].width)
        #print top[0].data.shape
        #sfsfd
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
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
      
        #softmax in dimension 0 
        max_cls = softmax(bottom[0].data,axis=0)
        top[0].data[0,:,0,0] = max_cls
        self.weights = weights(bottom[0].data,axis=0)
        if 0:
            import pylab
            pylab.figure(1)
            pylab.plot(max_cls)
            pylab.figure(2)
            pylab.imshow(self.weights)
            pylab.show()
            sdfsd
            raw_input()
        #print top[0].data.shape
        #fsf
        #cross entropy in dimension 0
        

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = top[0].diff.squeeze()*self.weights
        #print top[0].diff.squeeze()
        #raw_input()
        #print self.weights
        #print "Bottom",bottom[0].diff.sum(0).squeeze()
        
                
#softmax as normalized exp
class ExpSoftMaxLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if cfg.TRAIN.BETA==-1:
            layer_params = yaml.load(self.param_str_)
            self.beta = layer_params['beta']
        else:
            self.beta = cfg.TRAIN.BETA
        print "Beta",self.beta 
        #self.blobs.add_blob(1)
        #self.blobs[0].data[...] = 1 #beta
        #top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1],
        #        bottom[0].data.shape[2], bottom[0].data.shape[3])
        self.it=0
        #top[0].reshape(1, 21)
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
    #    pass
        #print "reshape"
        #top[0].reshape(1,21)
        top[0].reshape(bottom[0].num, bottom[0].channels)
        #        bottom[0].height, bottom[0].width)
        #top[0].reshape(1, bottom[0].channels,
        #        bottom[0].height, bottom[0].width)
        #top[0].reshape(bottom[0].num, bottom[0].channels,
        #        bottom[0].height, bottom[0].width)
        #top[0].reshape(1, bottom[0].channels,
        #        bottom[0].height, bottom[0].width)
        #print top[0].data.shape
        #sfsfd
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
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
        self.it+=1
        #beta=10
        p,ssum,entr = betaweights(bottom[0].data,self.beta,axis=0)
        top[0].data[...] = p
        #self.jacob = np.dot(p.T,p)
        #self.jacob = self.beta*(np.diag(p)-np.dot(p,p.T))/bottom[0].num
        #self.db = p/ssum*(bottom[0].data-)
        if 0:
            import pylab
            pylab.figure(1)
            pylab.plot(max_cls)
            pylab.figure(2)
            pylab.imshow(self.weights)
            pylab.show()
            sdfsd
            raw_input()
        #print top[0].data.shape
        #fsf
        #cross entropy in dimension 0
        

    def backward(self, top, propagate_down, bottom):
        #for l in range(21):
        #    bottom[0].diff[:,l] = top[0].data[:,l] * (top[0].diff[:,l]-np.dot(top[0].diff[:,l],top[0].data[:,l]))
        bottom[0].diff[...] = self.beta*top[0].data * (top[0].diff-(top[0].diff*top[0].data).sum(0,keepdims=True))/bottom[0].num
        #bottom[0].diff[...] = top[0].diff#np.dot(top[0].diff,top[0].data)
        #np.dot(top[0].diff,top[0].data)
        #if propagate_down[0]:
            #bottom[0].diff[...] = np.dot(top[0].diff,self.jacob)
            #self.blobs[0].diff=top[0].diff
        #print top[0].diff.squeeze()
        #raw_input()
        #print self.weights
        #print "Bottom",bottom[0].diff.sum(0).squeeze()


class BetaSoftMaxLayer(caffe.Layer):

    def setup(self, bottom, top):
        if cfg.TRAIN.BETA==-1:
            layer_params = yaml.load(self.param_str_)
            self.beta = layer_params['beta']
        else:
            self.beta = cfg.TRAIN.BETA
        print "Beta",self.beta 
        self.it=0
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels)

    def forward(self, bottom, top):
        self.it+=1
        p = betaweights2(bottom[0].data,self.beta,axis=0)
        top[0].data[...] = p
        if 0:
            import pylab
            pylab.figure(1)
            pylab.plot(max_cls)
            pylab.figure(2)
            pylab.imshow(self.weights)
            pylab.show()
            sdfsd
            raw_input()     

    def backward(self, top, propagate_down, bottom):
        #print "Size",bottom[0].num
        bottom[0].diff[...] = np.log(self.beta)*top[0].data * (top[0].diff-(top[0].diff*top[0].data).sum(0,keepdims=True))#/bottom[0].num
