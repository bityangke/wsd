# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import numpy as np
import caffe
from fast_rcnn.config import cfg
#from scipy.optimize import check_grad

def approx_fprime(px,f,epsilon=1e-4):
    xp=px.copy()
    y=f(px) 
    grad=np.zeros(list(px.shape),px.dtype)
    for x in xrange(px.shape[0]):
        #for y in xrange(px.shape[1]):
        #    for z in xrange(px.shape[2]):
        #        for h in xrange(px.shape[3]):
        xp[x,:]=xp[x,:]+epsilon
        fp=f(xp)
        xp[x,:]=xp[x,:]-2*epsilon
        fn=f(xp)
        xp[x,:]=xp[x,:]+epsilon
        grad[x,:]=(fp[0,:]-fn[0,:])/(2*epsilon)
        #print grad[x,0]
        #raw_input()
        #print grad[x,0]
    return grad

def check_grad(f,g,x,epsilon=1e-4):
    return np.sum((approx_fprime(x,f,epsilon)-g(x))**2)

def f(x):
    return x.sum(0,keepdims=True)
    
def g(x):
    return np.ones(x.shape,dtype=x.dtype)

class MySumLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        bottom[0].reshape(bottom[0].num, bottom[0].channels,bottom[0].height, bottom[0].width)
        top[0].reshape(1, bottom[0].channels,bottom[0].height, bottom[0].width)
        top[0].diff.reshape(1, bottom[0].channels,bottom[0].height, bottom[0].width)

    def forward(self, bottom, top):
        top[0].data[...]=f(bottom[0].data)

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            #bottom[0].diff[...] = top[0].diff.squeeze()*self.weights
            bottom[0].diff[...] = top[0].diff*g(bottom[0].data)
            
        if cfg.TRAIN.CHECK_GRAD:
            x = bottom[0].data
            err=check_grad(f,g,x,1e-4)
            print "Testing Gradient Sum",err
            if err>1e-3:
                dsfsd
                print "Error in the gradient!"
                
class MySum2Layer(caffe.Layer):
    #it receives also the rois to know which regions to sum

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Only two input needed.")
        self.num_im = cfg.TRAIN.IMS_PER_BATCH#bottom[1].data[:,0].max()
        #top[0].reshape(1, bottom[0].channels,bottom[0].height, bottom[0].width)
        
    def reshape(self, bottom, top):
        bottom[0].reshape(bottom[0].num, bottom[0].channels,bottom[0].height, bottom[0].width)
        top[0].reshape(cfg.TRAIN.IMS_PER_BATCH, bottom[0].channels,bottom[0].height, bottom[0].width)
        top[0].diff.reshape(cfg.TRAIN.IMS_PER_BATCH, bottom[0].channels,bottom[0].height, bottom[0].width)
        #print top[0].data.shape
        #print top[0].diff.shape
        #raw_input()
        #bottom[1][0] tells from which image every layer comes from

    def forward(self, bottom, top):
        for cl in xrange(self.num_im):
            sel = bottom[1].data[:,0]==cl
            top[0].data[cl]=f(bottom[0].data[sel])


    def backward(self, top, propagate_down, bottom):
        for cl in xrange(self.num_im):
            sel = bottom[1].data[:,0]==cl
            bottom[0].diff[sel] = top[0].diff[cl]*g(bottom[0].data[sel])
            
            if cfg.TRAIN.CHECK_GRAD:
                x = bottom[0].data[sel]
                err=check_grad(f,g,x,1e-4)
                print "Testing Gradient Sum",err
                if err>1e-3:
                    dsfsd
                    print "Error in the gradient!"

def betaweights2(x,beta,axis=-1):
    e_x = beta**(x - np.max(x,axis=axis,keepdims=True))
    #print e_x
    ssum=e_x.sum(axis=axis,keepdims=True)
    out = e_x / ssum
    #entr = np.sum(e_x*x,axis=axis,keepdims=True)
    return out

def approx_fprime_soft(px,f,epsilon=1e-4):
    xp=px.copy()
    y=f(px) 
    grad=np.zeros(list(px.shape),px.dtype)
    for x in xrange(px.shape[0]):
        #for y in xrange(px.shape[1]):
        #    for z in xrange(px.shape[2]):
        #        for h in xrange(px.shape[3]):
        xp[x,:]+=epsilon
        fp=f(xp)
        xp[x,:]-=2*epsilon
        fn=f(xp)
        xp[x,:]+=epsilon
        grad[x,:]=(fp[0,:]-fn[0,:])/(2*epsilon)
        #print grad[x,0]
    return grad

#def check_grad_soft(f_soft,g_soft,x,epsilon=1e-4):
#    return np.sum((approx_fprime_soft(x,f_soft,epsilon)-g_soft(x))**2)

def f_soft(x):
    return betaweights2(x,cfg.TRAIN.BETA,axis=0)
    
def g_soft(x):
    p = betaweights2(x,cfg.TRAIN.BETA,axis=0)
    return np.log(cfg.TRAIN.BETA)* p * (1-(p).sum(0,keepdims=True))
            
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
        #p = betaweights2(bottom[0].data,self.beta,axis=0)
        top[0].data[...] = f_soft(bottom[0].data)#p

    def backward(self, top, propagate_down, bottom):
        #bottom[0].diff[...] = np.log(self.beta)*top[0].data * (top[0].diff-(top[0].diff*top[0].data).sum(0,keepdims=True))#/bottom[0].num
        bottom[0].diff[...] = top[0].diff*g_soft(bottom[0].data)
        #print 'Error', np.sum((bottom[0].diff-pr2)**2)
        if cfg.TRAIN.CHECK_GRAD:
            x = bottom[0].data
            err=check_grad(f_soft,g_soft,x,1e-4)
            print "Testing Gradient Softmax",err
            if err>1e-3:
                print "Error in the softmax gradient!"
                sfdfs

class BetaSoftMax2Layer(caffe.Layer):

    def setup(self, bottom, top):
        if cfg.TRAIN.BETA==-1:
            layer_params = yaml.load(self.param_str_)
            self.beta = layer_params['beta']
        else:
            self.beta = cfg.TRAIN.BETA
        print "Beta",self.beta 
        self.it=0
        if len(bottom) != 2:
            raise Exception("Only two input needed.")
        self.num_im = cfg.TRAIN.IMS_PER_BATCH#bottom[1].data[:,0].max()

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels)

    def forward(self, bottom, top):
        self.it+=1
        #p = betaweights2(bottom[0].data,self.beta,axis=0)
        for cl in xrange(self.num_im):
            sel = bottom[1].data[:,0]==cl
            data_im = bottom[0].data[sel]
            top[0].data[sel] = f_soft(data_im)#p

    def backward(self, top, propagate_down, bottom):
        for cl in xrange(self.num_im):
            sel = bottom[1].data[:,0]==cl
            #for some reason g_soft is wrong...
            #bottom[0].diff[sel] = top[0].diff[sel]*g_soft(bottom[0].data[sel])
            bottom[0].diff[sel] = np.log(self.beta)*top[0].data[sel] * (top[0].diff[sel]-(top[0].diff[sel]*top[0].data[sel]).sum(0,keepdims=True))
        #print 'Error', np.sum((bottom[0].diff-pr2)**2)
            if cfg.TRAIN.CHECK_GRAD:
                x = bottom[0].data[sel]
                err=check_grad(f_soft,g_soft,x,1e-4)
                print "Testing Gradient Softmax",err
                if err>1e-3:
                    print "Error in the softmax gradient!"
                    sfdfs

