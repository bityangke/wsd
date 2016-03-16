import caffe
import numpy as np
import yaml
import os

from fast_rcnn.config import cfg

#deprecated: use the PlotSeg with the right parameters
class PlotSegAll(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        layer_params = yaml.load(self.param_str_)
        self.plot = layer_params['plot']
        self.savedir = os.path.join(cfg.ROOT_DIR,'output',cfg.EXP_DIR)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        print "Plotloss savedir",self.savedir
        self.loss=[0]
        self.it=0
        #layer_params = yaml.load(self.param_str_)
        #self._threshold = layer_params['threshold']
        #EXP_DIR
        #if len(bottom) != 2:
        #    raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        pass
        # check input dimensions match
        #if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        #top[0].reshape(1)

    def forward(self, bottom, top):
        #top[0].data=bottom[0].data
        if self.plot and self.it%100==0:
            import pylab
            import time
            #pylab.ion()
            #pylab.figure(1)
            #pylab.clf()
            #pylab.title("Image")
            #pylab.imshow(bottom[1].data[0,0])
            #pylab.draw()
            #pylab.show()
            pylab.figure(self.plot)
            pylab.clf()
            pylab.imshow(bottom[0].data[0,0],vmax=20,interpolation='nearest')
            pylab.draw()
            pylab.show()
            #print "max score class", bottom[0].data.max(2).max(2)
            #time.sleep(0.1)
            #pylab.ioff()
            raw_input()
        self.it+=1
           
        
    def backward(self, top, propagate_down, bottom):
        pass


class PlotLoss(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        layer_params = yaml.load(self.param_str_)
        self.plot = layer_params['plot']
        if layer_params.has_key('name'):
            self.name = layer_params['name']
        else:
            self.name = 'loss'
        self.savedir = os.path.join(cfg.ROOT_DIR,'output',cfg.EXP_DIR)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        print "Plotloss savedir",self.savedir
        self.loss=[0]
        self.it=0
        #layer_params = yaml.load(self.param_str_)
        #self._threshold = layer_params['threshold']
        #EXP_DIR
        #if len(bottom) != 2:
        #    raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        pass
        # check input dimensions match
        #if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        #top[0].reshape(1)

    def forward(self, bottom, top):
        batch=1
        size=1000#10000
        if self.it%(batch*size)==0:
            import cPickle
            cPickle.dump(self.loss,open(os.path.join(self.savedir,"%s.pkl"%self.name),'wr'))
            self.loss[-1]/=(size*batch)
            if self.plot:
                import pylab
                pylab.figure(1)
                pylab.clf()
                pylab.title("Loss")
                pylab.plot(self.loss)
                pylab.draw()
                pylab.show()
                raw_input()
            self.loss.append(0)
        
        if len(bottom[0].data.shape)==0:
            self.loss[-1]+=bottom[0].data
        else:
            self.loss[-1]+=bottom[0].data[0]
        self.it+=1
           
        
    def backward(self, top, propagate_down, bottom):
        pass


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            raw_input()
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            print "diff",self.diff.squeeze()
            raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num

def softmax(x,axis=-1):
    #e_x = np.exp(x - np.max(x,axis=axis))
    #out = e_x / e_x.sum(axis=axis)
    mmax = np.max(x,axis=axis,keepdims=True)
    #print mmax.shape,(x-mmax).shape,(np.log(np.sum(np.exp(x - mmax),axis=axis))).shape
    out = mmax.squeeze() + np.log(np.sum(np.exp(x - mmax),axis=axis))
    return out

def weights(x,axis=None):
    e_x = np.exp(x - np.max(x,axis=axis,keepdims=True))
    out = e_x / e_x.sum(axis=axis,keepdims=True)
    return out

#MP: assuming only one image per batch, otherwise normalize on number of images

class LogLoss(caffe.Layer):
    """
    Compute the Log Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #p = weights(bottom[0].data)
        self.diff[...] =  - bottom[1].data/(bottom[0].data+0.00001)
        top[0].data[...] = -np.sum(np.log(bottom[0].data)*bottom[1].data)#/bottom[0].count# / bottom[0].num / 2.
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",self.diff.squeeze()
            #raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count

class PlotSeg(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        layer_params = yaml.load(self.param_str_)
        self.plot = layer_params['plot']
        if layer_params.has_key('every'):
            self.every = layer_params['every']
        else:
            self.every = np.inf
        if layer_params.has_key('vmax'):
            self.vmax = layer_params['vmax']
        else:
            self.vmax = 255
        if layer_params.has_key('block'):
            self.block = layer_params['block']
        else:
            self.block = False
        self.savedir = os.path.join(cfg.ROOT_DIR,'output',cfg.EXP_DIR)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        print "Plotloss savedir",self.savedir
        self.loss=[0]
        self.it=0
        #layer_params = yaml.load(self.param_str_)
        #self._threshold = layer_params['threshold']
        #EXP_DIR
        #if len(bottom) != 2:
        #    raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        pass
        # check input dimensions match
        #if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        #top[0].reshape(1)

    def forward(self, bottom, top):
        #top[0].data=bottom[0].data
        if self.plot and self.it%self.every==0:
            import pylab
            if len(bottom)>2:
                cls=np.arange(bottom[2].data.shape[1])[bottom[2].data[0,:,0,0]!=0]
            else:
                #for hinge loss
                #cls=[(bottom[0].data[0]).reshape((bottom[0].data.shape[1],-1)).max(1).argmax()]
                #for softmax loss
                cls=[(bottom[0].data[0]).reshape((bottom[0].data.shape[1],-1)).sum(1).argmax()]
            for l in cls:
                #pylab.figure(1)
                #pylab.clf()
                #pylab.title("Image")
                #pylab.imshow(bottom[1].data[0,0])
                #pylab.draw()
                #pylab.show()
                pylab.figure(self.plot)
                pylab.clf()
                #pylab.title("Segment %s"%cls_names[l])
                #probability of being a certain class
                #pylab.imshow((bottom[0].data[0,l]),interpolation='nearest')
#                pylab.imshow(np.log(bottom[0].data[0,l]),interpolation='nearest')
                #if np.any(bottom[0].data[0,0]<20):
                #    vmax=20
                #else:
                #    vmax=20
                pylab.imshow(bottom[0].data[0,0],vmax=self.vmax,interpolation='nearest')
                #region with a certain class
                #pylab.imshow(bottom[0].data[0].argmax(0)==l,interpolation="nearest")
                pylab.draw()
                pylab.show()
                #print "max score class", bottom[0].data.max(2).max(2)
                #print "Min",bottom[0].data[0,l].min()
                #print "Max",bottom[0].data[0,l].max()
                if self.block:
                    #import time
                    #time.sleep(1)
                    raw_input()
        self.it+=1
           
        
    def backward(self, top, propagate_down, bottom):
        pass


class HingeLoss(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        layer_params = yaml.load(self.param_str_)
        if layer_params!=None:
            if layer_params.has_key('myloss_weight'):
                self.myloss_weight = layer_params['myloss_weight']
            else:
                self.myloss_weight = 1.0
        else:
            self.myloss_weight = 1.0
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #p = weights(bottom[0].data)
        bottom[1].data[bottom[1].data[...]>0]=1 #restore to a binary measure
        y = bottom[1].data*2-1
        act = 1-bottom[0].data*y
        self.diff=np.zeros(y.shape)
        self.diff[act>0] = - y[act>0] 
        top[0].data[...] = self.myloss_weight*np.sum(np.maximum(0,act))#/bottom[0].count# / bottom[0].num / 2.
        if np.any(np.isnan(act)):
            dsfsdf
        if 0:
            if self.myloss_weight>0:
                print "out",bottom[0].data.squeeze() 
                print "gt",bottom[1].data.squeeze()
                #print "p",p.squeeze()
                print "diff",self.diff.squeeze()
                print "Loss",top[0].data.squeeze()
                #raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = self.myloss_weight*sign * self.diff #/ bottom[i].count

class HingeLossLoc(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        #bottom[0] segmentation
        #bottom[1] score proposals
        #bottom[2] location proposals
        
        # check input pair
        self.norm= cfg.TRAIN.BRIDGE_NORM
        layer_params = yaml.load(self.param_str_)
        if layer_params.has_key('myloss_weight'):
            self.myloss_weight = layer_params['myloss_weight']
        else:
            self.myloss_weight = 1.0
        self.spatial_scale = layer_params['spatial_scale']
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[1].num != bottom[2].num:
            raise Exception("Score and proposals should have the same num")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        h=bottom[0].data.shape[2]
        w=bottom[0].data.shape[3]
        fbox = np.round(bottom[2].data[:,1:]*self.spatial_scale).astype(np.int)
        self.fbox = fbox
        #for each box
        loss=0
        fullneg=bottom[0].data.sum(3).sum(2)
        num_proposals = bottom[1].data.shape[0]
        num_classes = bottom[1].data.shape[1]
        self.act=np.zeros(num_proposals)
        self.hinge=np.zeros(num_proposals)
        pos = np.zeros((num_proposals,num_classes))
        neg = np.zeros((num_proposals,num_classes))
        for b in np.arange(num_proposals):
            #for the moment assume only 1 image per batch            
            wb=fbox[b,2]-fbox[b,0]
            hb=fbox[b,3]-fbox[b,1]
            if hb==0 or wb==0:
                pos[b]=0
                neg[b]=-1     
            elif h==hb and w==wb: #bounding box with size of the image
                pos[b]=0
                neg[b]=-1              
            else:
		if h*w-hb*wb==0:
		    sdfsd
                pos[b] = (bottom[0].data[0,:,fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1]).sum(2).sum(1)#/(hb*wb+1e-8)
                neg[b] = (fullneg-pos[b])/(h*w-hb*wb)
                pos[b] /= (hb*wb)
        self.act = neg-pos+1>0
        self.hinge = np.clip(neg-pos+1,0,np.inf)
        top[0].data[...] = self.myloss_weight*np.sum(bottom[1].data[:]*self.hinge)
        #if top[0].data[0]>100:
        #    sdssgd
        if 0:
            if self.myloss_weight>0:
                print "out",bottom[0].data.squeeze() 
                print "gt",bottom[1].data.squeeze()
                #print "p",p.squeeze()
                #print "diff",self.diff.squeeze()
                print "Loss",top[0].data.squeeze()
                #raw_input()

    def backward(self, top, propagate_down, bottom):
        h=bottom[0].data.shape[2]
        w=bottom[0].data.shape[3]
        num_proposals = bottom[1].data.shape[0]
        fbox = self.fbox
        bottom[0].diff[...] = 0
        for b in np.arange(num_proposals):
            if np.any(self.act[b]==True):
                wb=fbox[b,2]-fbox[b,0]
                hb=fbox[b,3]-fbox[b,1]
                if hb==0 or wb==0: #bounding box with size 0
                    pass
                elif h==hb and w==wb: #bounding box with size of the image
                    pass
                else:
                    bottom[0].diff[0,self.act[b],:,:] += bottom[1].data[b,self.act[b],np.newaxis,np.newaxis]/(h*w-hb*wb) #negative
                    bottom[0].diff[0,self.act[b],fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1] += -bottom[1].data[b,self.act[b],np.newaxis,np.newaxis]/(h*w-hb*wb) #negative
                    bottom[0].diff[0,self.act[b],fbox[b,1]:fbox[b,3]+1,fbox[b,0]:fbox[b,2]+1] += -bottom[1].data[b,self.act[b],np.newaxis,np.newaxis]/(hb*wb) #positive
        bottom[0].diff[...] *= self.myloss_weight
        bottom[1].diff[...] = self.myloss_weight*self.hinge
        if np.any(np.isnan(bottom[0].diff)):
            print "Nan error"
            dsfsf
        if np.any(np.isnan(bottom[1].diff)):
            print "Nan error"
            dsfsf


class HingeLoss2(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
#        if bottom[0].count != bottom[1].count:
#            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #p = weights(bottom[0].data)
        bottom[1].data[bottom[1].data[...]>0]=1 #restore to a binary measure
        y = bottom[1].data*2-1
        act = 1-bottom[0].data*y
        self.diff=np.zeros(y.shape)
        self.diff[act>0] = - y[act>0] 
        top[0].data[...] = np.sum(np.maximum(0,act))/bottom[0].num#/bottom[0].count# / bottom[0].num / 2.
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",self.diff.squeeze()
            print "Loss",top[0].data.squeeze()
            raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff/bottom[0].num #/ bottom[i].count

class MyDistance(caffe.Layer):
    #   bottom[0]: "fc7"       features
    #   bottom[1]: "soft_max"  weights
    #   bottom[2]: "rois"      image
    #   bottom[3]: "labels_im" image classes
    #it receives also the rois to know which regions to sum

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Only two input needed.")
        self.num_im = cfg.TRAIN.IMS_PER_BATCH#bottom[1].data[:,0].max()
        self.num_cl = bottom[3].data.shape[1]
        layer_params = yaml.load(self.param_str_)
        self.myloss_weight = layer_params['myloss_weight']
        
    def reshape(self, bottom, top):
        bottom[0].reshape(bottom[0].num, bottom[0].channels,bottom[0].height, bottom[0].width)
        top[0].reshape(1)#cfg.TRAIN.IMS_PER_BATCH, bottom[0].channels,bottom[0].height, bottom[0].width)

    def forward(self, bottom, top):
        self.acl=((bottom[3].data[0]+bottom[3].data[1])==2).squeeze()
        if cfg.TRAIN.SAME_CLASS_PAIR:
            assert(np.any(self.acl))
        cls=np.arange(self.num_cl)[self.acl]
        self.diff=np.zeros((self.num_cl,bottom[0].data.shape[1],1))
        self.dist=np.zeros(self.num_cl)
        for cl in cls: 
            aux=(bottom[0].data.T*bottom[1].data[:,cl]).T
            dsr0=(aux[bottom[2].data[:,0]==0]).sum(0)
            dsr1=(aux[bottom[2].data[:,0]==1]).sum(0)
            self.diff[cl] = (dsr0-dsr1).reshape((-1,1))
            self.dist[cl]=0.5*np.sum((self.diff)**2)
        top[0].data[0]=self.myloss_weight * self.dist.sum()/bottom[0].channels
        #print " DisLoss",top[0].data[0],
        

    def backward(self, top, propagate_down, bottom):
        #acl=(bottom[3].data[0]-bottom[3].data[1])==0
        bottom[0].diff[...]=0
        bottom[1].diff[...]=0
        if 1:#top[0].data[0]<0.1:
            cls=(np.arange(self.num_cl)[self.acl])
            for cl in cls:
                #bottom[0].diff[:,:,0,0] = bottom[0].diff[:,:,0,0] + self.myloss_weight * np.dot((bottom[1].data[:,cl]).reshape((-1,1)),self.diff[cl].T)
                #bottom[1].diff[:,cl] = bottom[1].diff[:,cl] + self.myloss_weight * (np.dot(bottom[0].data[:,:,0,0],self.diff[cl])[:,0])
                bottom[0].diff[:,:,0,0] += self.myloss_weight * np.dot((bottom[1].data[:,cl]).reshape((-1,1)),self.diff[cl].T)/bottom[0].channels
                bottom[1].diff[:,cl] += self.myloss_weight * (np.dot(bottom[0].data[:,:,0,0],self.diff[cl])[:,0])/bottom[0].channels
            sel = bottom[2].data[:,0]==1
            bottom[0].diff[sel] = -bottom[0].diff[sel]
            bottom[1].diff[sel] = -bottom[1].diff[sel]
                  
        if cfg.TRAIN.CHECK_GRAD:
            x = bottom[0].data[sel]
            err=check_grad(f,g,x,1e-4)
            print "Testing Gradient Sum",err
            if err>1e-3:
                dsfsd
                print "Error in the gradient!"

import math
class MyCosSim(caffe.Layer):
    #   bottom[0]: "fc7"       features
    #   bottom[1]: "soft_max"  weights
    #   bottom[2]: "rois"      image
    #   bottom[3]: "labels_im" image classes
    #it receives also the rois to know which regions to sum

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 4:
            raise Exception("Only two input needed.")
        self.num_im = cfg.TRAIN.IMS_PER_BATCH#bottom[1].data[:,0].max()
        self.num_cl = bottom[3].data.shape[1]
        layer_params = yaml.load(self.param_str_)
        self.myloss_weight = cfg.TRAIN.PAIRWISE_WEIGHT#layer_params['myloss_weight']
        print "1-Cos Loss weight",self.myloss_weight
        self.thr_trunc=0.3
        #top[0].reshape(1, bottom[0].channels,bottom[0].height, bottom[0].width)
        
    def reshape(self, bottom, top):
        bottom[0].reshape(bottom[0].num, bottom[0].channels,bottom[0].height, bottom[0].width)
        #top[0].reshape(bottom[0].num)
        top[0].reshape(1)#cfg.TRAIN.IMS_PER_BATCH, bottom[0].channels,bottom[0].height, bottom[0].width)
        #top[0].diff.reshape(cfg.TRAIN.IMS_PER_BATCH, bottom[0].channels,bottom[0].height, bottom[0].width)
        #print top[0].data.shape
        #print top[0].diff.shape
        #raw_input()
        #bottom[1][0] tells from which image every layer comes from

    def forward(self, bottom, top):
        self.acl=((bottom[3].data[0]+bottom[3].data[1])==2).squeeze()
        if cfg.TRAIN.SAME_CLASS_PAIR:
            assert(np.any(self.acl))
        cls=np.arange(self.num_cl)[self.acl]
        self.diff=np.zeros((self.num_cl,2,bottom[0].data.shape[1],1))
        self.dist=np.zeros(self.num_cl)
        for cl in cls: 
            #if truncated sample don't do anything
            #bestbb=bottom[1].data[:,cl]>self.thr_trunc
            aux=(bottom[0].data.T*bottom[1].data[:,cl]).T
            a=(aux[bottom[2].data[:,0]==0]).sum(0).squeeze()
            b=(aux[bottom[2].data[:,0]==1]).sum(0).squeeze()
            ab = np.dot(a,b)
            na = np.dot(a,a)+1e-8
            nb = np.dot(b,b)+1e-8
            sqab = math.sqrt(na)*math.sqrt(nb)
            self.diff[cl,0] = -(-a*ab/(na**2./3.*nb)+b/sqab).reshape((-1,1))
            self.diff[cl,1] = -(-b*ab/(nb**2./3.*na)+a/sqab).reshape((-1,1))
            self.dist[cl] = 1-ab/(sqab)
        top[0].data[0]=self.myloss_weight * self.dist.sum()#/bottom[0].channels
        #print " DisLoss",top[0].data[0],
        

    def backward(self, top, propagate_down, bottom):
        #acl=(bottom[3].data[0]-bottom[3].data[1])==0
        bottom[0].diff[...]=0
        bottom[1].diff[...]=0
        if 1:#top[0].data[0]<0.1:
            cls=(np.arange(self.num_cl)[self.acl])
            for cl in cls:
                #if truncated sample don't do anything
                sel = bottom[2].data[:,0]==0
                bottom[0].diff[sel,:,0,0] += self.myloss_weight * np.dot((bottom[1].data[sel,cl]).reshape((-1,1)),self.diff[cl,0].T)#/bottom[0].channels
                bottom[1].diff[sel,cl] += self.myloss_weight * (np.dot(bottom[0].data[sel,:,0,0],self.diff[cl,0])[:,0])#/bottom[0].channels
                sel = bottom[2].data[:,0]==1
                bottom[0].diff[sel,:,0,0] += self.myloss_weight * np.dot((bottom[1].data[sel,cl]).reshape((-1,1)),self.diff[cl,1].T)#/bottom[0].channels
                bottom[1].diff[sel,cl] += self.myloss_weight * (np.dot(bottom[0].data[sel,:,0,0],self.diff[cl,1])[:,0])#/bottom[0].channels
            
        #sel = bottom[2].data[:,0]==cl
        #bottom[0].diff[sel] = np.dot(bottom[1].data,top[0].diff[cl])
        
        if cfg.TRAIN.CHECK_GRAD:
            x = bottom[0].data[sel]
            err=check_grad(f,g,x,1e-4)
            print "Testing Gradient Sum",err
            if err>1e-3:
                dsfsd
                print "Error in the gradient!"


class HingeLossNorm(caffe.Layer):
    """
    Compute multiclass Hinge Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        #p = weights(bottom[0].data)
        bottom[1].data[bottom[1].data[...]>0]=1 #restore to a binary measure
        y = bottom[1].data*2-1
        act = 1-bottom[0].data*y
        self.diff=np.zeros(y.shape,dtype=np.float32)
        self.diff[act>0] = - y[act>0] 
        top[0].data[...] = np.sum(np.maximum(0,act))/bottom[0].channels# / bottom[0].num / 2.
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",self.diff.squeeze()
            print "Loss",top[0].data.squeeze()
            raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        #print "size",bottom[0].num,bottom[0].channels
        #print "Error",self.diff/ bottom[i].channels
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff/ bottom[i].channels

class SoftMaxLogLoss(caffe.Layer):
    """
    Compute the Log Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        #layer_params = yaml.load(self.param_str_)
        #self.plot = layer_params['plot']
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        p = weights(bottom[0].data)
        y = bottom[1].data/np.sum(bottom[1].data)
        self.diff[...] =  p - y
        top[0].data[...] = -np.sum(np.log(p)*y)
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",self.diff.squeeze()
            #raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count

class SoftMaxChanLogLoss(caffe.Layer):
    """
    Compute the Log Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        #layer_params = yaml.load(self.param_str_)
        #self.plot = layer_params['plot']
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        p = weights(bottom[0].data,1)
        y = bottom[1].data/np.sum(bottom[1].data,1)
        num_pix = bottom[0].height*bottom[0].width
        self.diff[...] =  (p - y)/(num_pix)
        top[0].data[...] = -np.sum(np.log(p)*y)/num_pix
        if 0:
            print "out",np.abs(bottom[0].data[0]).sum(2).sum(1)
            print "loss",top[0].data.squeeze()
            #print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",np.abs(self.diff[0]).sum(2).sum(1)
            #sfsd
            #raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count


class HingeChanLoss(caffe.Layer):
    """
    Compute the Log Loss 
    """

    def setup(self, bottom, top):
        # check input pair
        #layer_params = yaml.load(self.param_str_)
        #self.plot = layer_params['plot']
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        bottom[1].data[bottom[1].data[...]>0]=1 #restore to a binary measure
        y = bottom[1].data*2-1
        act = 1-bottom[0].data*y
        num_pix = bottom[0].height*bottom[0].width
        self.diff=np.zeros(y.shape,dtype=np.float32)/num_pix
        self.diff[act>0] = - y[act>0] 
        top[0].data[...] = np.sum(np.maximum(0,act))/num_pix# / bottom[0].num / 2.
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",self.diff.squeeze()
            print "Loss",top[0].data.squeeze()
            raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count


#class SoftMaxLogLoss(caffe.Layer):
#    """
#    Compute the Log Loss 
#    """
#
#    def setup(self, bottom, top):
#        # check input pair
#        if len(bottom) != 2:
#            raise Exception("Need two inputs to compute distance.")
#
#    def reshape(self, bottom, top):
#        # check input dimensions match
#        if bottom[0].count != bottom[1].count:
#            raise Exception("Inputs must have the same dimension.")
#        # difference is shape of inputs
#        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
#        # loss output is scalar
#        top[0].reshape(1)
#
#    def forward(self, bottom, top):
#        p = weights(bottom[0].data)
#        self.diff[...] =  p - bottom[1].data
#        top[0].data[...] = -np.sum(np.log(p)*bottom[1].data)
#        if 0:
#            print "out",bottom[0].data.squeeze() 
#            print "gt",bottom[1].data.squeeze()
#            #print "p",p.squeeze()
#            print "diff",self.diff.squeeze()
#            #raw_input()
#
#    def backward(self, top, propagate_down, bottom):
#        #print "propagate!!!",propagate_down[0],propagate_down[1]
#        for i in range(2):
#            if not propagate_down[i]:
#                continue
#            if i == 0:
#                sign = 1
#            else:
#                sign = -1
#            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count


class SoftMaxLogCurrLoss(caffe.Layer):
    """
    Compute the Log Loss 
    """

    def setup(self, bottom, top):
        self.it=0
        layer_params = yaml.load(self.param_str_)
        self._threshold = layer_params['threshold']
        print "USING CURRICULUM!!!!"
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)
        self.it+=1

    def forward(self, bottom, top):
        p = weights(bottom[0].data)
        self.diff[...] =  p - bottom[1].data
        top[0].data[...] = -np.sum(np.log(p)*bottom[1].data)
        if 0:
            print "out",bottom[0].data.squeeze() 
            print "gt",bottom[1].data.squeeze()
            #print "p",p.squeeze()
            print "diff",self.diff.squeeze()
            #raw_input()

    def backward(self, top, propagate_down, bottom):
        #print "propagate!!!",propagate_down[0],propagate_down[1]
        for i in range(2):
            #print "loss",top[0].data
            mythr=max(7.0-0.0001*self.it,self._threshold)
            if self.it%100==0:
            #mythr=self._threshold#2.5#max(self._threshold-0.001*self.it,1.0)
                print "Thr",mythr#,"it",self.it
            #if top[0].data[0]>self._threshold:
            #    print "Too difficult sample, not updating!"
            if top[0].data>mythr:
                #print "Too difficult sample, not updating!"
                bottom[i].diff[...]=0
                continue
            #else:
                #print "Updating!"
            if not propagate_down[i]:# and top[0].data[0]>self._threshold:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count
