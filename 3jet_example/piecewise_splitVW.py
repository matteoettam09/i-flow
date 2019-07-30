import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from tensorflow.keras import layers, models, activations

class PiecewiseLinear(tfb.Bijector):
    """
    Piecewise Linear: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseLinear"):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseLinear, self).__init__(
                forward_min_event_ndims=1, validate_args=validate_args, name=name,
        )
        self.D, self.d = D, d
        self.id = layer_id
        self.nbins = nbins
        self.width = 1.0/self.nbins
        self.range = tf.cast(tf.range(self.d),dtype=tf.int32)
        self.QMat = self.buildQ(self.d, self.nbins)
        self.trainable_variables = self.QMat.trainable_variables

    def buildQ(self, d, nbins):
        inval = layers.Input(shape=(d,))
        
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
        
        
        out = layers.Dense((self.D-d)*nbins)(h8)
        out = layers.Reshape(((self.D-d),nbins))(out)
        model = models.Model(inval,out)
        model.summary()
        return model

    def Q(self, xd):
        QMat = tf.nn.softmax(self.QMat(xd),axis=-1)
        return QMat

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
        #return -self._inverse_log_det_jacobian(self._forward(x))
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
    

class PiecewiseQuadratic(tfb.Bijector):
    """
    Piecewise Quadratic: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, nchannels=1, validate_args=False, name="PiecewiseQuadratic"):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseQuadratic, self).__init__(
                forward_min_event_ndims=1, validate_args=validate_args, name=name,
        )

        self.D, self.d = D, d
        self.id = layer_id
        self.nbins = nbins
        self.nchannels=nchannels
        self.range = tf.range(self.d)
        self.WMat = self.buildW(self.d, self.nbins)
        self.VMat = self.buildV(self.d, self.nbins)
#        self.NNMat = self.buildNN()
        self.trainable_variables = [
                self.WMat.trainable_variables,
                self.VMat.trainable_variables,
        ]

    def buildW(self, d, nbins):
        inval = layers.Input(shape=(d*self.nchannels#*nbins
                                    ,))
        h1 = layers.Dense(32,activation='relu')(inval)
        h2 = layers.Dense(32,activation='relu')(h1)
        out = layers.Dense((self.D-d)*nbins)(h2)
        out = layers.Reshape((self.D-d,nbins))(out)
        out = activations.softmax(out,axis=-1)
        model = models.Model(inval,out)
        return model

    def buildV(self, d, nbins):
        inval = layers.Input(shape=(d*self.nchannels#*nbins
                                    ,))
        #d1 = layers.Dense(32)(inval)
        #a1 = activations.relu(d1)
        #d2 = layers.Dense(16)(a1)
        #a2 = activations.relu(d2)
        #d3 = layers.Dense(8,activation='relu')(a2)
        #d4 = layers.Dense(8)(d3)
        #c1 = layers.concatenate([d4,d2],axis=-1)
        #a3 = activations.relu(c1)
        #d5 = layers.Dense(16)(a3)
        #c2 = layers.concatenate([d5,d1], axis=-1)
        #d6 = layers.Dense((self.D-d)*(nbins+1),#activation=self.sigmoidC)(c2)
        h1 = layers.Dense(32,activation='relu')(inval)
        h2 = layers.Dense(32,activation='relu')(h1)
        out = layers.Dense((self.D-d)*(nbins+1))(h2)
        out = layers.Reshape((self.D-d,nbins+1))(out)
        model = models.Model(inval,out)
        return model
#
#    def W(self, xd):
#        WMat = tf.nn.softmax(self.WMat(xd),axis=-1)
#        return WMat
#
#    def V(self, xd, W):
#        VMat = self.VMat(xd)
#        VExp = tf.exp(VMat)
#        VSum = tf.reduce_sum((VExp[...,0:self.nbins]+VExp[...,1:self.nbins+1])*W[...,:self.nbins]/2,axis=-1,keepdims=True)
#        VMat = tf.truediv(VExp,VSum)
#        return VMat

    def sigmoidC(self,x):
        return 5.0/(1.0+tf.exp(-x))

  #  def buildNN(self):
   #     inval = layers.Input(shape=(self.d#*self.nbins
   #                                 *self.nchannels,))
        
        #h1 = layers.Dense(256,activation='relu')(inval)
        #h2 = layers.Dense(128,activation='relu')(h1)
        #h3 = layers.Dense(64,activation='relu')(h2)
        #h4 = layers.Dense(32,activation='relu')(h3)
        #h5 = layers.Dense(32,activation='relu')(h4)
        #h5 = layers.concatenate([h5,h3], axis=-1)
        #h6 = layers.Dense(64,activation='relu')(h5)
        #h6 = layers.concatenate([h6,h2], axis=-1)
        #h7 = layers.Dense(128,activation='relu')(h6)
        #h7 = layers.concatenate([h7,h1], axis=-1)
        #h8 = layers.Dense(256,activation='relu')(h7)        
        #out = layers.Dense((self.D-self.d)*(2*self.nbins+1),activation='relu')(h8)
        
    #    d1 = layers.Dense((self.D-self.d)*(2*self.nbins+1))(inval)
     #   a1 = activations.relu(d1)
      #  d2 = layers.Dense((self.D-self.d)*(2*self.nbins+1)//2)(a1)
      #  a2 = activations.relu(d2)
      #  d3 = layers.Dense((self.D-self.d)*(2*self.nbins+1)//4,activation='relu')(a2)
      #  d4 = layers.Dense((self.D-self.d)*(2*self.nbins+1)//4)(d3)
      #  c1 = layers.concatenate([d4,d2],axis=-1)
      #  a3 = activations.relu(c1)
      #  d5 = layers.Dense((self.D-self.d)*(2*self.nbins+1)//2)(a3)
      #  c2 = layers.concatenate([d5,d1], axis=-1)
      #  d6 = layers.Dense((self.D-self.d)*(2*self.nbins+1),activation=self.sigmoidC
      #  )(c2)
        #out = layers.Reshape(((self.D-self.d),2,self.nbins+1))(out)
        #out = activations.softmax(out,axis=-1)
      #  out = layers.Reshape(((self.D-self.d),2*self.nbins+1))(d6)
      #  model = models.Model(inval,out)
      #  model.summary()
      #  return model

    def one_blob(self, xd):
        y = tf.tile(((0.5/self.nbins) + tf.range(0.,1.,delta = 1./self.nbins)),[tf.size(xd)]) 
        y = tf.reshape(y,(-1,self.d,self.nbins))
        res = tf.exp(((-self.nbins*self.nbins)/2.)*(y-xd[...,tf.newaxis])**2)
        return res    

    def GetWV(self, xd,channel):
        channel_hot = tf.one_hot(tf.cast(channel,dtype=tf.int32),depth=self.nchannels, axis=-1)
        One_blob = tf.reshape(xd,[-1,self.d])
       ## one blob encoding:
        #One_blob = tf.reshape(self.one_blob(xd),[-1,self.d*self.nbins])
        channel_blob = channel_hot[...,None] * One_blob[:,None]
        channel_blob = tf.reshape(channel_blob,(-1,self.d#*self.nbins
                                                *self.nchannels))
       # NNMat = self.NNMat(channel_blob)
       # NNMat = self.NNMat(xd)
        #W = tf.nn.softmax(NNMat[...,:self.nbins],axis=-1)
        W = self.WMat(channel_blob)
        #W = NNMat[...,:self.nbins]
       # W = tf.where(tf.less(W,1e-6*tf.ones_like(W)),1e-6*tf.ones_like(W),W)
       # W = W+tf.log(1e-10)
        W = tf.truediv(W,tf.reduce_sum(W,axis=-1,keepdims=True))
       # V = NNMat[...,self.nbins:]
        V = self.VMat(channel_blob)
        V = tf.exp(V)
        VSum = tf.reduce_sum((V[...,:self.nbins]+V[...,1:])*W*0.5,axis=-1,keepdims=True)
        V = tf.truediv(V,VSum)
        #V = tf.where(tf.greater(V,1e3*tf.ones_like(V)),1e3*tf.ones_like(V),V)
       # V = tf.where(tf.less(V,1e-8*tf.ones_like(V)),1e-8*tf.ones_like(V),V)
       # V = tf.truediv(V,tf.reduce_sum((V[...,:self.nbins]+V[...,1:])*W*0.5,axis=-1,keepdims=True))
        return W, V


    def _find_bins(self,x,y):
        ibins = tf.cast(tf.searchsorted(y,x[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.where(tf.equal(ibins,self.nbins*tf.ones_like(ibins)),ibins-1,ibins)
        ibins = tf.reshape(ibins,[tf.shape(x)[0],self.D-self.d])
        #mask=x[...,tf.newaxis]-y>1e-6
        #y=tf.cast(mask,tf.float32)*y
        #ibins=tf.argmax(y,axis=-1)+tf.cast(tf.reduce_max(y,axis=-1)>0,tf.int64)
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        one_hot_sum = tf.one_hot(ibins-1,depth=self.nbins)
        one_hot_V = tf.one_hot(ibins,depth=self.nbins+1)

        return one_hot, one_hot_sum, one_hot_V

    def pdf(self,x,channel):
        xd, xD = x[..., :self.d], x[..., self.d:]
        W, V = self.GetWV(xd,channel)
        WSum = tf.cumsum(W,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD,WSum)
        Wb = tf.reduce_sum(W*one_hot,axis=-1)
        alpha = tf.where(tf.equal(tf.zeros_like(Wb),Wb),
                         tf.zeros_like(xD),
                         (xD-tf.reduce_sum(WSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(Wb))
        result = tf.reduce_sum((V[...,1:]-V[...,:-1])*one_hot,axis=-1)*alpha \
                +tf.reduce_sum(V*one_hot_V,axis=-1)
        return tf.concat([xd, result], axis=-1) 

    def _inverse(self, x,channel=None): #forward
        "Calculate forward coupling layer"
        xd, xD = x[..., :self.d], x[..., self.d:]
        W, V = self.GetWV(xd,channel)
        WSum = tf.cumsum(W,axis=-1)
        VSum = tf.cumsum((V[...,1:]+V[...,:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD,WSum)
        alpha = (xD-tf.reduce_sum(WSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        Wb = tf.reduce_sum(W*one_hot,axis=-1)
        alpha = tf.where(tf.equal(tf.zeros_like(Wb),Wb),
                         tf.zeros_like(xD),
                         (xD-tf.reduce_sum(WSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(Wb))
        yD = alpha**2/2*tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1) \
                *tf.reduce_sum(W*one_hot,axis=-1) \
                + alpha*tf.reduce_sum(V*one_hot_V,axis=-1)*tf.reduce_sum(W*one_hot,axis=-1) \
                + tf.reduce_sum(VSum*one_hot_sum,axis=-1)
        return tf.concat([xd, yD], axis=-1)

    def _forward(self, y,channel): #inverse
        "Calculate inverse coupling layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        W, V = self.GetWV(yd,channel)
        WSum = tf.cumsum(W,axis=-1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD,VSum)
        Wb = tf.reduce_sum(W*one_hot,axis=-1)
        beta = tf.where(tf.equal(tf.zeros_like(Wb),Wb),
                         tf.zeros_like(yD),
                         (yD-tf.reduce_sum(VSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(Wb))
        denom = tf.where(tf.equal(tf.zeros_like(Wb),Wb),
                         tf.zeros_like(yD),
                         tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1))
        Vbins = tf.reduce_sum(V*one_hot_V,axis=-1)
        xD = tf.where(tf.equal(tf.zeros_like(denom),denom),
                tf.div_no_nan(yD-tf.reduce_sum(VSum*one_hot_sum,axis=-1),Vbins)\
                      + tf.reduce_sum(WSum*one_hot_sum,axis=-1),
                tf.div_no_nan((-Vbins+tf.sqrt(Vbins**2+2*beta*denom)),denom)*Wb\
                      + tf.reduce_sum(WSum*one_hot_sum,axis=-1)
                      )
       # xD = Wb*xD + tf.reduce_sum(WSum*one_hot_sum,axis=-1)
        return tf.concat([yd, xD], axis=-1)

    def _inverse_log_det_jacobian(self, x,channel): #forward
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x,channel)[...,self.d:]),axis=-1)

    def _forward_log_det_jacobian(self, y,channel): #inverse
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        W, V = self.GetWV(yd,channel)
        WSum = tf.cumsum(W,axis=1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD,VSum)
        Wb = tf.reduce_sum(W*one_hot,axis=-1)
        beta = tf.where(tf.equal(tf.zeros_like(Wb),Wb),
                        tf.zeros_like(yD),
                         (yD-tf.reduce_sum(VSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(Wb))
        denom = tf.where(tf.equal(tf.zeros_like(Wb),Wb),
                        tf.zeros_like(yD),
                         tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)
                         )
        Vbins = tf.reduce_sum(V*one_hot_V,axis=-1)
        alpha =  tf.div_no_nan((-Vbins+tf.sqrt(Vbins**2+2*beta*denom)),denom)
        result = tf.where(tf.equal(tf.zeros_like(denom),denom),
                          Vbins,
                          tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)*alpha+Vbins)
        return -tf.reduce_sum(tf.log(result),axis=-1)

    

class PiecewiseQuadraticConst(tfb.Bijector):
    """
    Piecewise Quadratic with constant bin widths: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseQuadraticConst"):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseQuadraticConst, self).__init__(
                forward_min_event_ndims=1, validate_args=validate_args, name=name,
        )

        self.D, self.d = D, d
        self.id = layer_id
        self.nbins = nbins
        self.range = tf.range(self.d)
#        self.WMat = self.buildW(self.d, self.nbins)
        self.VMat = self.buildV(self.d, self.nbins)
#        self.NNMat = self.buildNN(self.d, self.nbins)
        self.trainable_variables = [
#                self.NNMat.trainable_variables,
                self.VMat.trainable_variables,
        ]

    def buildW(self, d, nbins):
#        inval = layers.Input(shape=(d,))
#        h1 = layers.Dense(16,activation='relu')(inval)
#        h2 = layers.Dense(16,activation='relu')(h1)
#        out = layers.Dense(d*nbins,activation='relu')(h2)
#        out = layers.Reshape((d,nbins))(out)
#        model = models.Model(inval,out)
#        return model
        inval = layers.Input(shape=(d,))
        out = layers.Dense((self.D-d)*nbins,activation='relu')(inval)
        out = layers.Reshape((self.D-d,nbins))(out)
        out = layers.Lambda(lambda x: (1./nbins)*tf.ones_like(x))(out)
        model = models.Model(inval,out)
        return model

    def buildV(self, d, nbins):
        inval = layers.Input(shape=(d,))
        h1 = layers.Dense(128,activation='relu')(inval)
        h2 = layers.Dense(64,activation='relu')(h1)
        h3 = layers.Dense(32,activation='relu')(h2)
        h4 = layers.Dense(32,activation='relu')(h3)
        h4 = layers.concatenate([h4,h2], axis=-1)
        h5 = layers.Dense(64,activation='relu')(h4)
        h5 = layers.concatenate([h5,h1], axis=-1)
        h6 = layers.Dense(128,activation='relu')(h5)
        out = layers.Dense((self.D-d)*(nbins+1),activation='relu')(h6)
        out = layers.Reshape(((self.D-d),nbins+1))(out)
        model = models.Model(inval,out)
        model.summary()
        return model

    def W(self, xd):
        #WMat = tf.nn.softmax(self.WMat(xd),axis=-1)
        #return WMat
        return tf.constant(1./self.nbins,shape=(np.shape(xd)[0],self.D-self.d,self.nbins))

    def V(self, xd, W):
        VMat = self.VMat(xd)
        VExp = tf.exp(VMat)
        VSum = tf.reduce_sum((VExp[...,0:self.nbins]+VExp[...,1:self.nbins+1])*W[...,:self.nbins]/2,axis=-1,keepdims=True)
        VMat = tf.truediv(VExp,VSum)
        return VMat

#    def buildNN(self, d, nbins):
#        inval = layers.Input(shape=(d,))
#        h1 = layers.Dense(64,activation='relu')(inval)
#        h2 = layers.Dense(64,activation='relu')(h1)
#        out = layers.Dense(d*(2*nbins+1),activation='relu')(h2)
#        out = layers.Reshape((d,2*nbins+1))(out)
#        model = models.Model(inval,out)
#        model.summary()
#        return model

#    def GetWV(self, xd):
#        NNMat = self.NNMat(xd)
#        W = tf.nn.softmax(NNMat[...,:self.nbins],axis=-1)
#        V = NNMat[...,self.nbins:]
#        VExp = tf.exp(V)
#        VSum = tf.reduce_sum((VExp[...,:self.nbins]+VExp[...,1:])*W/2,axis=-1,keepdims=True)
#        V = tf.truediv(VExp,VSum)
#        return W, V

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
#        W, V = self.GetWV(xd)
        W = self.W(xd)
        V = self.V(xd, W)
        WSum = tf.cumsum(W,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD,WSum)
        alpha = (xD-tf.reduce_sum(WSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        result = tf.reduce_sum((V[...,1:]-V[...,:-1])*one_hot,axis=-1)*alpha \
                +tf.reduce_sum(V*one_hot_V,axis=-1)
        return tf.concat([xd, result], axis=-1) 

    def _forward(self, x): #forward
        "Calculate forward coupling layer"
        xd, xD = x[..., :self.d], x[..., self.d:]
#        W, V = self.GetWV(xd)
        W = self.W(xd)
        V = self.V(xd, W)

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

    def _inverse(self, y): #inverse
        "Calculate inverse coupling layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
#        W, V = self.GetWV(yd)
        W = self.W(yd)
        V = self.V(yd, W)

        WSum = tf.cumsum(W,axis=-1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD,VSum)
        denom = tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)
        beta = (yD - tf.reduce_sum(VSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        Vbins = tf.reduce_sum(V*one_hot_V,axis=-1)
        xD = tf.where(tf.equal(tf.zeros_like(denom),denom),
                beta/Vbins,
                tf.div_no_nan(1.,denom)*(-Vbins+tf.sqrt(Vbins**2+2*beta*denom))
        )
        xD = tf.reduce_sum(W*one_hot,axis=-1)*xD + tf.reduce_sum(WSum*one_hot_sum,axis=-1)
#        xD = tf.where(tf.is_nan(xD), tf.ones_like(xD), xD)
        return tf.concat([yd, xD], axis=-1)

    def _forward_log_det_jacobian(self, x): #forward
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[...,self.d:]),axis=-1)

    def _inverse_log_det_jacobian(self, y): #inverse
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
#        W, V = self.GetWV(yd)
        W = self.W(yd)
        V = self.V(yd, W)
        WSum = tf.cumsum(W,axis=1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD,VSum)
        denom = tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)
        beta = (yD - tf.reduce_sum(VSum*one_hot_sum,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(W*one_hot,axis=-1))
        Vbins = tf.reduce_sum(V*one_hot_V,axis=-1)
        alpha = tf.where(tf.equal(tf.zeros_like(denom),denom),
        #alpha = tf.where(tf.less_equal(1e-6*tf.ones_like(denom),tf.abs(denom)),
                beta/Vbins,
                tf.div_no_nan(1.,denom)*(-Vbins+tf.sqrt(Vbins**2+2*beta*denom))
        )
        result = tf.reduce_sum((V[...,1:]-V[...,0:-1])*one_hot,axis=-1)*alpha+Vbins
        return -tf.reduce_sum(tf.log(result),axis=-1)

# Tests if run using main
if __name__ == '__main__':
    nevents = 1000
    NP_DTYPE=np.float32

    print('Testing Linear')
    with tf.Session() as sess:
        # Piecewise Linear Tests
        testLinear = PiecewiseLinear(6,3,5)
        val = np.array(np.random.rand(nevents,6),dtype=NP_DTYPE)
        forward = testLinear.forward(val)
        inverse = testLinear.inverse(forward)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Preform calculations
        print(np.allclose(val,sess.run(inverse)))

        forward_jacobian = testLinear.forward_log_det_jacobian(val,event_ndims=1)
        inverse_jacobian = testLinear.inverse_log_det_jacobian(forward,event_ndims=1)
        difference = forward_jacobian+inverse_jacobian
        diff_result = sess.run(difference)
        print(np.allclose(diff_result,np.zeros_like(diff_result)))

    print('Testing Quadratic')
    with tf.Session() as sess:
        # Piecewise Quadratic Tests
        testQuadratic = PiecewiseQuadratic(6,3,5)
        val = np.array(np.random.rand(nevents,6),dtype=NP_DTYPE)
        forward = testQuadratic.forward(val)
        inverse = testQuadratic.inverse(forward)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Preform calculations
        print(np.allclose(val,sess.run(inverse)))

        forward_jacobian = testQuadratic.forward_log_det_jacobian(val,event_ndims=1)
        inverse_jacobian = testQuadratic.inverse_log_det_jacobian(forward,event_ndims=1)
        difference = forward_jacobian+inverse_jacobian
        diff_result = sess.run(difference)
        print(np.allclose(diff_result,np.zeros_like(diff_result),atol=1e-6))

    print('Testing Distribution')
    with tf.Session() as sess:
        test_bijector = PiecewiseLinear(2,1,5,layer_id=0)
        base_dist = tfd.Uniform(low=2*[0.0],high=2*[1.0])
        dist = tfd.TransformedDistribution(
                distribution=base_dist,
                bijector=test_bijector,
                event_shape=[1],
        ) 

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        print(sess.run(dist.sample(5)))

