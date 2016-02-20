import caffe
import numpy as np
import yaml
import os

from fast_rcnn.config import cfg

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


class PlotLoss(caffe.Layer):
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
        batch=1
        size=1000#10000
        if self.it%(batch*size)==0:
            import cPickle
            cPickle.dump(self.loss,open(os.path.join(self.savedir,"loss.pkl"),'wr'))
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

class HingeLoss(caffe.Layer):
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
        self.diff=np.zeros(y.shape)
        self.diff[act>0] = - y[act>0] 
        top[0].data[...] = np.sum(np.maximum(0,act))#/bottom[0].count# / bottom[0].num / 2.
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
        top[0].data[...] = np.sum(np.maximum(0,act))#/bottom[0].count# / bottom[0].num / 2.
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
        layer_params = yaml.load(self.param_str_)
        self.plot = layer_params['plot']
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
        self.diff[...] =  p - bottom[1].data
        top[0].data[...] = -np.sum(np.log(p)*bottom[1].data)
        if self.plot:
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

class SoftMaxLogLoss(caffe.Layer):
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
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff #/ bottom[i].count


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
