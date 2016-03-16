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

class ScoreSegmBoxesLayer(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        #bottom[0] segmentation
        #bottom[1] softmax segmentation
        #bottom[2] location proposals
        
        self.beta=cfg.TRAIN.BETA
        # check input pair
        layer_params = yaml.load(self.param_str_)
        if layer_params.has_key('myloss_weight'):
            self.myloss_weight = layer_params['myloss_weight']
        else:
            self.myloss_weight = 1.0
        self.spatial_scale = layer_params['spatial_scale']
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance.")
        if layer_params.has_key('mode'):
            self.mode=layer_params['mode']
        else:
            self.mode="softmax"
        if layer_params.has_key('mul'):
            self.mul=layer_params['mul']
        else:
            self.mul=1.0

    def reshape(self, bottom, top):
        self.num_proposals = bottom[2].data.shape[0]
        self.num_classes = bottom[1].data.shape[1]
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Score and proposals should have the same num")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(self.num_proposals,self.num_classes)
        #sdfsd

    def forward(self, bottom, top):
        #self.num_proposals = bottom[2].data.shape[0]
        #self.num_classes = bottom[2].data.shape[1]
        #top[0].reshape(self.num_proposals,self.num_classes)
        h=bottom[0].data.shape[2]
        w=bottom[0].data.shape[3]
        fbox = np.round(bottom[2].data[:,1:]*self.spatial_scale).astype(np.int)
        #fbox=fbox.clip([0,0,0,0],[w,h,w,h])
        self.fbox = fbox
        self.pix_scores = []
        self.pix_prob = []
        #for each box
        if self.mode=='mean-neg':
            #padding
            padb = -1*np.ones((bottom[0].data.shape[1],h+2,w+2),dtype=bottom[0].data.dtype)
            padb[:,1:-1,1:-1]=bottom[0].data[0]
        for b in np.arange(self.num_proposals):
            #for the moment assume only 1 image per batch            
            wb=fbox[b,2]-fbox[b,0]
            hb=fbox[b,3]-fbox[b,1]
            assert(wb<=w and hb<=h)
            if hb==0 or wb==0:
                top[0].data[b] = -np.inf
                self.pix_scores.append(None)
                self.pix_prob.append(None)
            else:
                pix_scores = bottom[0].data[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1]
                if self.mode=='softmax':
                    pix_prob = betaweights2((bottom[1].data[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1]).reshape((self.num_classes,-1)),self.beta,1).reshape((self.num_classes,hb+1,wb+1))
                elif self.mode=='mean':
                    pix_prob = np.ones((self.num_classes,hb+1,wb+1))/((hb+1)*(wb+1))#bottom[1].data[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1]
                elif self.mode=='mean-neg':
                    pix_scores = -padb[:,fbox[b,1]:fbox[b,3]+3,fbox[b,0]:fbox[b,2]+3]
                    pix_scores[:,1:-1,1:-1] = 2*padb[:,fbox[b,1]+1:fbox[b,3]+2,fbox[b,0]+1:fbox[b,2]+2]
                    pix_prob = np.ones((self.num_classes,hb+3,wb+3))/float(2*(hb+2)+2*(wb+2))
                    pix_prob[:,1:-1,1:-1] = 1/float((hb+1)*(wb+1))
                    #note: backward not implemented
                elif self.mode=='sum':
                    pix_prob = np.ones((self.num_classes,hb+1,wb+1))#bottom[1].data[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1]
                self.pix_scores.append(pix_scores)
                self.pix_prob.append(pix_prob)
                top[0].data[b] = (pix_scores*pix_prob).sum(2).sum(1)*self.mul
                if np.any(np.isnan(pix_scores)) or np.any(np.isnan(pix_prob)) or np.any(np.isnan(top[0].data[b])):
                    errorfsd
                if 0:
                    print "box",b,"score",top[0].data[b]
        #raw_input()


    def backward(self, top, propagate_down, bottom):
        h=bottom[0].data.shape[2]
        w=bottom[0].data.shape[3]
        #num_proposals = bottom[1].data.shape[0]
        fbox = self.fbox
        bottom[0].diff[...] = 0
        bottom[1].diff[...] = 0
        for b in np.arange(self.num_proposals):
            wb=fbox[b,2]-fbox[b,0]
            hb=fbox[b,3]-fbox[b,1]
            assert(wb<=w and hb<=h)
            if hb==0 or wb==0 or np.any(top[0].data[b]==-np.inf):
                continue
            else:                
                bottom[0].diff[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1] += self.pix_prob[b]*top[0].diff[b][:,np.newaxis,np.newaxis]*self.mul#bottom[1].data[0,:,fbox[b,0]:fbox[b,2]+1,fbox[b,1]:fbox[b,3]+1]*top[0].diff[b]
                if self.mode=='softmax':                    
                    diff = top[0].diff[b][:,np.newaxis,np.newaxis] * self.pix_scores[b]
                    aux = np.log(self.beta)*self.pix_prob[b] * (diff-(diff*self.pix_prob[b]).sum(0,keepdims=True))
                    bottom[1].diff[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1] += aux*self.mul
                else:
                    pass
                    #bottom[1].diff[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1] += self.pix_scores[b]*top[0].diff[b][:,np.newaxis,np.newaxis]
                #bottom[1].diff[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1] = self.pix_scores[b]*np.log(self.beta)*self.pix_prob[b] * (top[0].diff[b][:,np.newaxis,np.newaxis]-(top[0].diff[b][:,np.newaxis,np.newaxis]*self.pix_prob[b]).sum(0,keepdims=True))
        #np.log(self.beta)*top[0].data * (top[0].diff-(top[0].diff*top[0].data).sum(0,keepdims=True))
                if np.any(np.isnan(bottom[0].diff)) or np.any(np.isnan(bottom[1].diff)):
                    errorfsd
        if 0:
            print "diff1",bottom[0].diff[0,1].squeeze() 
            print "diff2",bottom[1].diff[0,1].squeeze()
            print "topdiff",top[0].diff[0,1].squeeze()
            #rsdffsd

    
class MySoftMaxLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(1, bottom[0].channels,
                bottom[0].height, bottom[0].width)
        
    def forward(self, bottom, top):
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
     
    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = top[0].diff.squeeze()*self.weights

class MyDummyLayer(caffe.Layer):
    #used to not vbisualize milions of outputs...

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        if len(top)>0:
            top[0].reshape(bottom[0].num,bottom[0].channels,bottom[0].height,bottom[0].width)
        #top[0].reshape(1)
        if 0:
            print bottom[0].data
            raw_input()
        
    def forward(self, bottom, top):
        if len(top)>0:
            top[0].data[...] = bottom[0].data
        if 0:
            print "Forward"
            print bottom[0].data[0].max(1).max(1)
            #raw_input()

    def backward(self, top, propagate_down, bottom):
        if 0:
            print "Backward"
            print top[0].diff
            raw_input()

class MySkipBg(caffe.Layer):
    #used to not vbisualize milions of outputs...

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        if len(top)>0:
            top[0].reshape(bottom[0].num,bottom[0].channels,bottom[0].height,bottom[0].width)
        #top[0].reshape(1)
        if 0:
            print bottom[0].data
            raw_input()
        
    def forward(self, bottom, top):
        if len(top)>0:
            top[0].data[...] = bottom[0].data
            top[0].data[:,0] = -np.inf
        if 0:
            print "Forward"
            print bottom[0].data[0].max(1).max(1)
            #raw_input()

    def backward(self, top, propagate_down, bottom):
        if 0:
            print "Backward"
            print top[0].diff
            raw_input()

class MyArgMaxLayer(caffe.Layer):
    #used to not vbisualize milions of outputs...

    def setup(self, bottom, top):
        # check input pair
        self.it = 0
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        #bottom[0].data.argmax(1)[np.newaxis]
        top[0].reshape(bottom[0].num,1,bottom[0].height,bottom[0].width)
        if 0:
            print bottom[0].data.shape
            raw_input()
        
    def forward(self, bottom, top):
        #top[0].data[...]=2
        top[0].data[...]=bottom[0].data.argmax(1)[np.newaxis]
        if 0:
            print "max score class", bottom[0].data.max(2).max(2)
            #print "Scores",bottom[0].data[:,:,0:4,0:4]
            #print "Argmax",bottom[0].data[:,:,0:4,0:4].argmax(1)
            #print "ArgMax",top[0].data[:,:,0:4,0:4]
            #if self.it>0:
            #    dsfdsfs
            self.it += 1
            #raw_input()
            
        #top[0]=np.argmax(bottom[0].data,1,keepdims=True)
        #top[0].data[...] = np.sum(bottom[0].data)

    def backward(self, top, propagate_down, bottom):
        pass

class SparseInputLayer(caffe.Layer):
    #used to not vbisualize milions of outputs...

    def setup(self, bottom, top):
        # check input pair
        self.it = 0
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        #bottom[0].data.argmax(1)[np.newaxis]
        top[0].reshape(bottom[0].num,bottom[0].channels,bottom[0].height,bottom[0].width)
        
    def forward(self, bottom, top):
        top[0].data[...]=-1
        cls = np.arange(bottom[0].channels)[bottom[0].data.squeeze()!=0]
        top[0].data[0,:len(cls),0,0]=cls
        if 0:
            print "Sparse input:",top[0].data.squeeze()

    def backward(self, top, propagate_down, bottom):
        pass

class DenseInputLayer(caffe.Layer):
    #used to not vbisualize milions of outputs...

    def setup(self, bottom, top):
        # check input pair
        self.it = 0
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        #bottom[0].data.argmax(1)[np.newaxis]
        top[0].reshape(bottom[0].num,21,bottom[0].height,bottom[0].width)
        
    def forward(self, bottom, top):
        top[0].data[...]=0
        for c in range(21):
            aux = (bottom[0].data[0,0]==c)
            top[0].data[0,c,aux]=1
        if 0:
            print "Dense input:",top[0].data.squeeze()

    def backward(self, top, propagate_down, bottom):
        pass

class ConstrainedLayer(caffe.Layer):
    #used to not vbisualize milions of outputs...

    def setup(self, bottom, top):
        # check input pair
        self.it = 0
        if len(bottom) != 2:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        #bottom[0].data.argmax(1)[np.newaxis]
        top[0].reshape(bottom[0].num,bottom[0].channels,bottom[0].height,bottom[0].width)
        
    def forward(self, bottom, top):
        top[0].data[...]=-np.inf#bottom[0].data
        for c in range(bottom[1].channels):
            if bottom[1].data[0,c]==-1:
                continue
            else:
                lab=int(bottom[1].data[0,c,0,0])
                top[0].data[0,lab]=bottom[0].data[0,lab]
        if 0:
            print "Dense input:",top[0].data.squeeze()

    def backward(self, top, propagate_down, bottom):
        pass

        
class MySoftMaxConvLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        #top[0].reshape(bottom[0].num, bottom[0].channels,
        #        bottom[0].height, bottom[0].width)
        top[0].reshape(bottom[0].num, bottom[0].channels,
                1, 1)
        
    def forward(self, bottom, top):
        #softmax in dimension 0 
        aux=bottom[0].data.reshape((bottom[0].num, bottom[0].channels,
            -1))
        #print "SMax",aux.max(2)
        #print "SMin",aux.min(2)
        top[0].data[:,:,0,0] = softmax(aux,axis=2)
        self.weights = weights(aux,axis=2).reshape((bottom[0].num, bottom[0].channels,
                bottom[0].height, bottom[0].width))   

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            bottom[0].diff[...] = top[0].diff*self.weights

class MyExpConvLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels,
                bottom[0].height, bottom[0].width)
        
    def forward(self, bottom, top):
        #softmax in dimension 0 
        aux=bottom[0].data.reshape((bottom[0].num, bottom[0].channels,
            -1))
        top[0].data = np.exp(bottom[0].data,axis=2).reshape((bottom[0].num, bottom[0].channels,
                bottom[0].height, bottom[0].width))        

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
            aux = top[0].diff.reshape((bottom[0].num, bottom[0].channels,
            -1))
            bottom[0].diff[...] = np.exp(aux,axis=2).reshape((bottom[0].num, bottom[0].channels,
                bottom[0].height, bottom[0].width))

class MyMaxLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(1, bottom[0].channels,
                bottom[0].height, bottom[0].width)
        
    def forward(self, bottom, top):
        #softmax in dimension 0 
        top[0].data[0,:,0,0] = bottom[0].data.max(0)
        self.argmax = bottom[0].data.argmax(0)
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
        #for idcl,cl in enumerate(self.argmax):
        bottom[0].diff[self.argmax,np.arange(self.argmax.shape[0])] = top[0].diff.squeeze()
        
                
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

class BetaSoftMaxConvLayer(caffe.Layer):

    def setup(self, bottom, top):
        #if cfg.TRAIN.BETA==-1:
        #    layer_params = yaml.load(self.param_str_)
        #    self.beta = layer_params['beta']
        #else:
        #    self.beta = cfg.TRAIN.BETA
        layer_params = yaml.load(self.param_str_)
        if layer_params!=None and layer_params.has_key('beta'):
            self.beta = layer_params['beta']
        else:
            self.beta = cfg.TRAIN.BETA
        print "Beta",self.beta 
        self.it=0
        if len(bottom) != 1:
            raise Exception("Only one input needed.")

    def reshape(self, bottom, top):
        top[0].reshape(bottom[0].num, bottom[0].channels,bottom[0].height, bottom[0].width)

    def forward(self, bottom, top):
        self.it+=1
        aux = bottom[0].data.reshape((bottom[0].num, bottom[0].channels,-1))
        p = betaweights2(aux,self.beta,axis=2)
        top[0].data[...] = p.reshape((bottom[0].num, bottom[0].channels,bottom[0].height, bottom[0].width))
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
        
        bottom[0].diff[...] = np.log(self.beta)*top[0].data * (top[0].diff-((top[0].diff*top[0].data).sum(2,keepdims=True).sum(3,keepdims=True)))#/bottom[0].num

