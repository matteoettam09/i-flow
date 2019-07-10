import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow.keras import layers, models

class Piecewise(tfb.Bijector):
    def __init__(self,D,d,nbins,blob=False,hot=False,model=None,unet=False,**kwargs):
        super(Piecewise, self).__init__(**kwargs)
        self.D, self.d = D, d
        self.nbins = nbins
        self.range = tf.cast(tf.range(self.d),dtype=tf.int32)
        self.blob = blob
        self.hot = hot
        self.model = model
        self.unet = unet

    def one_blob(self, xd):
        y = tf.tile(((0.5*self.width) + tf.range(0.,1.,delta = 1./self.nbins)),[tf.size(xd)]) 
        y = tf.reshape(y,(-1,self.d,self.nbins))
        res = tf.exp(((-self.nbins*self.nbins)/2.)*(y-xd[...,tf.newaxis])**2)
        return res

    def one_hot(self, xd):
        ibins = tf.cast(tf.floor(xd*self.nbins),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        one_hot = tf.one_hot(ibins,depth=self.nbins, axis=-1)
        one_hot = tf.reshape(one_hot,[-1,self.d*self.nbins])
        return one_hot

    def build_input(self):
        if self.blob:
            inval = layers.Input(shape=(self.d*self.nbins,))
        if self.hot:
            inval = layers.Input(shape=(self.d*self.nbins,))
        else:
            inval = layers.Input(shape=(self.d,))

        return inval

    def build_unet(self,inval):
        h1 = layers.Dense(256,activation='relu')(inval)
        h2 = layers.Dense(128,activation='relu')(h1)
        h3 = layers.Dense(64,activation='relu')(h2)
        h4 = layers.Dense(32,activation='relu')(h3)
        h5 = layers.Dense(32,activation='relu')(h4)
        h5 = layers.concatenate([h5,h3], axis=-1)
        h6 = layers.Dense(64,activation='relu')(h5)
        h6 = layers.concatenate([h6,h2], axis=-1)
        h7 = layers.Dense(128,activation='relu')(h6)
        h7 = layers.concatenate([h7,h1], axis=-1)
        h8 = layers.Dense(256,activation='relu')(h7)        

        return h8

    def build_dense(self,inval):
        h1 = layers.Dense(128,activation='relu')(inval)
        h2 = layers.Dense(128,activation='relu')(h1)
        h3 = layers.Dense(128,activation='relu')(h2)
        h4 = layers.Dense(128,activation='relu')(h3)

        return h4

class PiecewiseLinear(Piecewise):
    """
    Piecewise Linear: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseLinear", **kwargs):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseLinear, self).__init__(D, d, nbins,
                forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs
        )
        self.id = layer_id
        self.width = 1.0/self.nbins
        self.QMat = self.buildQ(self.d, self.nbins)
        self.trainable_variables = self.QMat.trainable_variables

    def buildQ(self, d, nbins):
        inval = self.build_input()

        if self.model is not None:
            if isinstance(self.model,models.Model):
                model = self.model
                model.summary()
                return model
            else:
                print('WARNING: Passed in network is not valid. Defaulting to Unet.')
                unet = self.build_unet(inval)
                out = layers.Dense((self.D-d)*nbins)(unet)
        elif self.unet:
            unet = self.build_unet(inval)
            out = layers.Dense((self.D-d)*nbins)(unet)
        else:
            dense = self.build_dense(inval)
            out = layers.Dense((self.D-d)*nbins)(dense)

        out = layers.Reshape(((self.D-d),nbins))(out)
        model = models.Model(inval,out)
        model.summary()
        return model
    
    def Q(self, xd):
        if self.hot:
            xd = self.one_hot(xd)
        elif self.blob:
            xd = tf.reshape(self.one_blob(xd),[-1,self.d*self.nbins])
        
        return tf.nn.softmax(self.QMat(xd),axis=-1)

    def pdf(self,x):
        xd, xD = x[..., :self.d], x[..., self.d:]
        Q = self.Q(xd)
        ibins = tf.cast(tf.floor(xD*self.nbins),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        return tf.concat([tf.ones_like(xd), tf.reduce_sum(Q*one_hot,axis=-1)/self.width], axis=-1)
        
    def _inverse(self, x): #forward
        "Calculate forward coupling layer"
        xd, xD = x[..., :self.d], x[..., self.d:]
        Q = self.Q(xd)
        ibins = tf.cast(tf.floor(xD*self.nbins),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        one_hot2 = tf.one_hot(ibins-1,depth=self.nbins)
        yD = ((xD*self.nbins-tf.cast(ibins,dtype=tf.float32)) \
           * tf.reduce_sum(Q*one_hot,axis=-1)) \
           + tf.reduce_sum(tf.cumsum(Q,axis=-1)*one_hot2,axis=-1)
        return tf.concat([xd, yD], axis=-1)

    def _forward(self, y): #inverse
        "Calculate inverse coupling layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        Q = self.Q(yd)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q,axis=-1),yD[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        ibins = tf.reshape(ibins,[tf.shape(yD)[0],self.D-self.d])
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        one_hot2 = tf.one_hot(ibins-1,depth=self.nbins)
        xD = ((yD-tf.reduce_sum(tf.cumsum(Q,axis=-1)*one_hot2,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(Q*one_hot,axis=-1)) \
                +tf.cast(ibins,dtype=tf.float32))*self.width
        return tf.concat([yd, xD], axis=-1)

    def _inverse_log_det_jacobian(self, x): #forward
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[...,self.d:]),axis=-1)

    def _forward_log_det_jacobian(self, y): #inverse
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        Q = self.Q(yd)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q,axis=-1),yD[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        ibins = tf.reshape(ibins,[tf.shape(yD)[0],self.D-self.d])
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        return -tf.reduce_sum(tf.log(tf.reduce_sum(Q*one_hot,axis=-1)/self.width),axis=-1)

class PiecewiseQuadratic(Piecewise):
    """
    Piecewise Quadratic: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseQuadratic",**kwargs):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseQuadratic, self).__init__(D, d, nbins,
                forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs
        )

        self.id = layer_id
        self.NNMat = self.buildNN()
        self.trainable_variables = self.NNMat.trainable_variables

    def buildNN(self):
        inval = self.build_input()

        if self.model is not None:
            if isinstance(self.model,models.Model):
                model = self.model
                model.summary()
                return model
            else:
                print('WARNING: Passed in network is not valid. Defaulting to Unet.')
                unet = self.build_unet(inval)
                out = layers.Dense((self.D-self.d)*(2*self.nbins+1),activation='relu')(unet)
        elif self.unet:
            unet = self.build_unet(inval)
            out = layers.Dense((self.D-self.d)*(2*self.nbins+1),activation='relu')(unet)
        else:
            dense = self.build_dense(inval)
            out = layers.Dense((self.D-self.d)*(2*self.nbins+1),activation='relu')(dense)

        out = layers.Reshape(((self.D-self.d),2*self.nbins+1))(out)
        model = models.Model(inval,out)
        model.summary()
        return model

    def GetWV(self, xd):
        if self.hot:
            xd = self.one_hot(xd)
        elif self.blob:
            xd = tf.reshape(self.one_blob(xd),[-1,self.d*self.nbins])

        NNMat = self.NNMat(xd)
        W = tf.nn.softmax(NNMat[...,:self.nbins],axis=-1)
        W = tf.where(tf.less(W,1e-6*tf.ones_like(W)),1e-6*tf.ones_like(W),W)
        V = NNMat[...,self.nbins:]
        VExp = tf.exp(V)
        VSum = tf.reduce_sum((VExp[...,:self.nbins]+VExp[...,1:])*W/2,axis=-1,keepdims=True)
        V = tf.truediv(VExp,VSum)
        return W, V

    def _find_bins(self,x,y):
        ibins = tf.cast(tf.searchsorted(y,x[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        ibins = tf.reshape(ibins,[tf.shape(x)[0],self.D-self.d])
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        one_hot_sum = tf.one_hot(ibins-1,depth=self.nbins)
        one_hot_V = tf.one_hot(ibins,depth=self.nbins+1)

        return one_hot, one_hot_sum, one_hot_V

    def pdf(self,x):
        xd, xD = x[..., :self.d], x[..., self.d:]
        W, V = self.GetWV(xd)
        WSum = tf.cumsum(W,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD,WSum)
        alpha = (xD-tf.reduce_sum(WSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        result = tf.reduce_sum((V[...,1:]-V[...,:-1])*one_hot,axis=-1)*alpha \
                +tf.reduce_sum(V*one_hot_V,axis=-1)
        return tf.concat([xd, result], axis=-1) 

    def _inverse(self, x): #forward
        "Calculate forward coupling layer"
        xd, xD = x[..., :self.d], x[..., self.d:]
        W, V = self.GetWV(xd)
        WSum = tf.cumsum(W,axis=-1)
        VSum = tf.cumsum((V[...,1:]+V[...,:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD,WSum)
        alpha = (xD-tf.reduce_sum(WSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        yD = alpha**2/2*tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1) \
                *tf.reduce_sum(W*one_hot,axis=-1) \
                + alpha*tf.reduce_sum(V*one_hot_V,axis=-1)*tf.reduce_sum(W*one_hot,axis=-1) \
                + tf.reduce_sum(VSum*one_hot_sum,axis=-1)
        return tf.concat([xd, yD], axis=-1)

    def _forward(self, y): #inverse
        "Calculate inverse coupling layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        W, V = self.GetWV(yd)
        WSum = tf.cumsum(W,axis=-1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD,VSum)
        denom = tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)
        beta = (yD - tf.reduce_sum(VSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        Vbins = tf.reduce_sum(V*one_hot_V,axis=-1)
        xD = tf.where(tf.equal(tf.zeros_like(denom),denom),
                beta/Vbins,
                tf.div_no_nan((-Vbins+tf.sqrt(Vbins**2+2*beta*denom)),denom)
        )
        xD = tf.reduce_sum(W*one_hot,axis=-1)*xD + tf.reduce_sum(WSum*one_hot_sum,axis=-1)
        return tf.concat([yd, xD], axis=-1)

    def _inverse_log_det_jacobian(self, x): #forward
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[...,self.d:]),axis=-1)

    def _forward_log_det_jacobian(self, y): #inverse
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        W, V = self.GetWV(yd)
        WSum = tf.cumsum(W,axis=1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD,VSum)
        denom = tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)
        beta = (yD - tf.reduce_sum(VSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        Vbins = tf.reduce_sum(V*one_hot_V,axis=-1)
        alpha = tf.where(tf.equal(tf.zeros_like(denom),denom),
                beta/Vbins,
                tf.div_no_nan((-Vbins+tf.sqrt(Vbins**2+2*beta*denom)),denom)
        )
        result = tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)*alpha+Vbins
        return -tf.reduce_sum(tf.log(result),axis=-1)

class PiecewiseQuadraticConst(PiecewiseQuadratic):
    """
    Piecewise Quadratic with constant bin widths: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseQuadraticConst",**kwargs):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseQuadratic, self).__init__(D, d, nbins,
                forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs
        )

        self.id = layer_id
        self.VMat = self.buildV()
        self.trainable_variables = self.VMat.trainable_variables

    def buildV(self):
        inval = self.build_input()

        if self.model is not None:
            if isinstance(self.model,models.Model):
                model = self.model
                model.summary()
                return model
            else:
                print('WARNING: Passed in network is not valid. Defaulting to Unet.')
                unet = self.build_unet(inval)
                out = layers.Dense((self.D-self.d)*(2*self.nbins+1),activation='relu')(unet)
        elif self.unet:
            unet = self.build_unet(inval)
            out = layers.Dense((self.D-self.d)*(self.nbins+1),activation='relu')(unet)
        else:
            dense = self.build_dense(inval)
            out = layers.Dense((self.D-self.d)*(self.nbins+1),activation='relu')(dense)

        out = layers.Reshape(((self.D-self.d),self.nbins+1))(out)
        model = models.Model(inval,out)
        model.summary()
        return model
      
    def W(self, xd):
        return tf.constant(1./self.nbins,shape=(np.shape(xd)[0],self.D-self.d,self.nbins))

    def V(self, xd, W):
        if self.hot:
            xd = self.one_hot(xd)
        elif self.blob:
            xd = tf.reshape(self.one_blob(xd),[-1,self.d*self.nbins])

        VMat = self.VMat(xd)
        VExp = tf.exp(VMat)
        VSum = tf.reduce_sum((VExp[...,0:self.nbins]+VExp[...,1:self.nbins+1])*W[...,:self.nbins]/2,axis=-1,keepdims=True)
        VMat = tf.truediv(VExp,VSum)
        return VMat

    def GetWV(self, xd):
        W = self.W(xd)
        return W, self.V(xd, W)