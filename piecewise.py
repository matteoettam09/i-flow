import numpy as np
import tensorflow as tf
tfd = tf.contrib.distributions
tfb = tfd.bijectors
from tensorflow.keras import layers, models

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
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.D, self.d = D, d
        self.id = layer_id
        self.nbins = nbins
        self.width = 1.0/self.nbins
        self.range = np.arange(self.d)
        self.QMat = self.buildQ(self.d, self.nbins)
        self.trainable_variables = self.QMat.trainable_variables
        
    def buildQ(self, d, nbins):
        inval = layers.Input(shape=(d,))
        h1 = layers.Dense(64,activation='relu')(inval)
        h2 = layers.Dense(64,activation='relu')(h1)
        out = layers.Dense(d*nbins,activation='relu')(h2)
        out = layers.Reshape((d,nbins))(out)
        model = models.Model(inval,out)
        model.summary()
        return model
        
    def Q(self, xd):
        QMat = tf.nn.softmax(self.QMat(xd),axis=2)
        QMat = np.insert(QMat,0,0,axis=2)
        return QMat
        
    def pdf(self,x):
        xd, xD = x[:, :self.d], x[:, self.d:]
        Q = self.Q(xd)
        ibins = np.array(np.floor(xD*self.nbins),dtype=np.int32)+1
        grid = np.array(np.meshgrid(np.arange(len(xD)),self.range)).T
        bins = tf.concat([grid,ibins[np.arange(len(xD)),:,np.newaxis]],axis=2)
        return tf.concat([xd, tf.gather_nd(Q,bins)/self.width], axis=1)
        
    def _forward(self, x):
        "Calculate forward coupling layer"
        xd, xD = x[:, :self.d], x[:, self.d:]
        Q = self.Q(xd)
        ibins = np.array(np.floor(xD*self.nbins),dtype=np.int32)
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(xD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(xD)),:,np.newaxis]],axis=2)
        binsSum = tf.concat([grid,ibins[np.arange(len(xD)),:,np.newaxis]],axis=2)
        yD = (xD*self.nbins-ibins)*tf.gather_nd(Q,bins)+tf.gather_nd(np.cumsum(Q,axis=2),binsSum)
        return tf.concat([xd, yD], axis=1)
        
    def _inverse(self, y):
        "Calculate inverse coupling layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        Q = self.Q(yd)
        ibins = tf.searchsorted(np.cumsum(Q,axis=2),yD[...,np.newaxis],side='right')
        ibins = ibins.numpy().reshape(len(yD),self.d)
        ibins = ibins-1
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(yD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(yD)),:,np.newaxis]],axis=2)
        binsSum = tf.concat([grid,ibins[np.arange(len(yD)),:,np.newaxis]],axis=2)
        xD = ((yD-tf.gather_nd(np.cumsum(Q,axis=2),binsSum))*tf.reciprocal(tf.gather_nd(Q,bins))\
              +np.array(ibins,dtype=np.float32))*self.width
        xD = tf.where(tf.is_nan(xD), tf.ones_like(xD), xD)
        return tf.concat([yd, xD], axis=1)
    
    def _forward_log_det_jacobian(self, x):
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[:,self.d:]),axis=-1)
    
    def _inverse_log_det_jacobian(self, y):
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        Q = self.Q(yd)
        ibins = tf.searchsorted(np.cumsum(Q,axis=2),yD[...,np.newaxis],side='right')
        ibins = ibins.numpy().reshape(len(yD),self.d)
        ibins = ibins-1
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(yD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(yD)),:,np.newaxis]],axis=2)
        return -tf.reduce_sum(tf.log(tf.gather_nd(Q,bins)/self.width),axis=-1)

class PiecewiseQuadratic(tfb.Bijector):
    """
    Piecewise Quadratic: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseLinear"):
        """
        Args:
            D: number of dimensions
            d: First d units are pass-thru units.
        """
        super(PiecewiseQuadratic, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name
        )
        self.D, self.d = D, d
        self.id = layer_id
        self.nbins = nbins
        self.range = np.arange(self.d)
        self.WMat = self.buildW(self.d, self.nbins)
        self.VMat = self.buildV(self.d, self.nbins)
        self.trainable_variables = [
            self.WMat.trainable_variables,
            self.VMat.trainable_variables,
        ]
        
    def buildW(self, d, nbins):
        inval = layers.Input(shape=(d,))
        h1 = layers.Dense(64,activation='relu')(inval)
        h2 = layers.Dense(64,activation='relu')(h1)
        out = layers.Dense(d*nbins,activation='relu')(h2)
        out = layers.Reshape((d,nbins))(out)
        model = models.Model(inval,out)
        return model
    
    def buildV(self, d, nbins):
        inval = layers.Input(shape=(d,))
        h1 = layers.Dense(64,activation='relu')(inval)
        h2 = layers.Dense(64,activation='relu')(h1)
        out = layers.Dense(d*(nbins+1),activation='relu')(h2)
        out = layers.Reshape((d,nbins+1))(out)
        model = models.Model(inval,out)
        return model
        
    def W(self, xd):
        WMat = tf.nn.softmax(self.WMat(xd),axis=2)
        WMat = np.insert(WMat,0,0,axis=2)
        return WMat
              
    def V(self, xd, W):
        VMat = self.VMat(xd)
        VExp = tf.exp(VMat)
        VSum = np.sum((VExp[...,0:self.nbins]+VExp[...,1:self.nbins+1])*W[...,1:self.nbins+1]/2,axis=2,keepdims=True)
        VMat = np.true_divide(VExp,VSum)
        VMat = np.insert(VMat,0,0,axis=2)
        return VMat
        
    def pdf(self,x):
        xd, xD = x[:, :self.d], x[:, self.d:]
        W = self.W(xd)
        V = self.V(xd,W)
        WSum = np.cumsum(W,axis=2)
        ibins = tf.searchsorted(WSum,xD[...,np.newaxis],side='right')
        ibins = ibins.numpy().reshape(len(xD),self.d)
        ibins = ibins-1
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(xD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(xD)),:,np.newaxis]],axis=2)
        binsSum = tf.concat([grid,ibins[np.arange(len(xD)),:,np.newaxis]],axis=2)
        alpha = (xD-tf.gather_nd(WSum,binsSum))*tf.reciprocal(tf.gather_nd(W,bins))
        result = tf.gather_nd(np.diff(V),bins)*alpha+tf.gather_nd(V,bins)
        return tf.concat([xd, result], axis=1) 
        
    def _forward(self, x):
        "Calculate forward coupling layer"
        xd, xD = x[:, :self.d], x[:, self.d:]
        W = self.W(xd)
        V = self.V(xd,W)
        WSum = np.cumsum(W,axis=2)
        VSum = np.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=2)
        ibins = tf.searchsorted(WSum,xD[...,np.newaxis],side='right')
        ibins = ibins.numpy().reshape(len(xD),self.d)
        ibins = ibins-1
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(xD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(xD)),:,np.newaxis]],axis=2)
        binsSum = tf.concat([grid,ibins[np.arange(len(xD)),:,np.newaxis]],axis=2)
        alpha = (xD-tf.gather_nd(WSum,binsSum))/tf.gather_nd(W,bins)
        yD = alpha**2/2*tf.gather_nd(np.diff(V),bins)*tf.gather_nd(W,bins) \
           + alpha*tf.gather_nd(V,bins)*tf.gather_nd(W,bins) \
           + tf.gather_nd(VSum,binsSum)
        return tf.concat([xd, yD], axis=1)
    
    def _inverse(self, y):
        "Calculate inverse coupling layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        W = self.W(yd)
        V = self.V(yd,W)
        WSum = np.cumsum(W,axis=2)
        VSum = np.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=2)
        ibins = tf.searchsorted(VSum,yD[...,np.newaxis],side='right')
        ibins = ibins.numpy().reshape(len(yD),self.d)
        ibins = ibins-1
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(yD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(yD)),:,np.newaxis]],axis=2)
        binsSum = tf.concat([grid,ibins[np.arange(len(yD)),:,np.newaxis]],axis=2)
        denom = tf.gather_nd(np.diff(V),bins)
        beta = (yD - tf.gather_nd(VSum,binsSum))/tf.gather_nd(W,bins)
        Vbins = tf.gather_nd(V,bins)
        xD = tf.where(tf.equal(tf.zeros_like(denom),denom),
                      beta/Vbins,
                      1/denom*(-Vbins+tf.sqrt(Vbins**2+2*beta*denom))
                      )
        xD = tf.gather_nd(W,bins)*xD + tf.gather_nd(WSum,binsSum)
        xD = tf.where(tf.is_nan(xD), tf.ones_like(xD), xD)
        return tf.concat([yd, xD], axis=1)
    
    def _forward_log_det_jacobian(self, x):
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[:,self.d:]),axis=-1)
    
    def _inverse_log_det_jacobian(self, y):
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        W = self.W(yd)
        V = self.V(yd,W)
        WSum = np.cumsum(W,axis=1)
        VSum = np.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=2)
        ibins = tf.searchsorted(VSum,yD[...,np.newaxis],side='right')
        ibins = ibins.numpy().reshape(len(yD),self.d)
        ibins = ibins-1
        ibinsp1 = ibins+1
        grid = np.array(np.meshgrid(np.arange(len(yD)),self.range)).T
        bins = tf.concat([grid,ibinsp1[np.arange(len(yD)),:,np.newaxis]],axis=2)
        binsSum = tf.concat([grid,ibins[np.arange(len(yD)),:,np.newaxis]],axis=2)
        denom = tf.gather_nd(np.diff(V),bins)
        beta = (yD - tf.gather_nd(VSum,binsSum))/tf.gather_nd(W,bins)
        Vbins = tf.gather_nd(V,bins)
        alpha = tf.where(tf.equal(tf.zeros_like(denom),denom),
                      beta/Vbins,
                      1/denom*(-Vbins+tf.sqrt(Vbins**2+2*beta*denom))
                      )
        result = tf.gather_nd(np.diff(V),bins)*alpha+Vbins
        return -tf.reduce_sum(tf.log(result),axis=-1)

# Tests if run using main
if __name__ == '__main__':
    nevents = 2000000
    NP_DTYPE=np.float32
    tf.enable_eager_execution()

    # Piecewise Linear Tests
    testLinear = PiecewiseLinear(6,3,5)
    val = np.array(np.random.rand(nevents,6),dtype=NP_DTYPE)
    forward = testLinear.forward(val)
    inverse = testLinear.inverse(forward)
    print(np.allclose(val,inverse.numpy()))

    forward_jacobian = testLinear.forward_log_det_jacobian(val,event_ndims=1)
    inverse_jacobian = testLinear.inverse_log_det_jacobian(forward,event_ndims=1)
    difference = forward_jacobian+inverse_jacobian
    print(np.allclose(forward_jacobian,-inverse_jacobian))
    print((difference/forward_jacobian).numpy()[np.logical_not(np.isclose(forward_jacobian,-inverse_jacobian))])
    print(len((difference/forward_jacobian).numpy()[np.logical_not(np.isclose(forward_jacobian,-inverse_jacobian))])/nevents)

    # Piecewise Quadratic Tests
    testQuadratic = PiecewiseQuadratic(6,3,5)
    val = np.array(np.random.rand(nevents,6),dtype=NP_DTYPE)
    forward = testQuadratic.forward(val)
    inverse = testQuadratic.inverse(forward)
    print(np.allclose(val,inverse.numpy()))
    print(((val-inverse.numpy())/val)[np.logical_not(np.isclose(val,inverse.numpy()))])

    forward_jacobian = testQuadratic.forward_log_det_jacobian(val,event_ndims=1)
    inverse_jacobian = testQuadratic.inverse_log_det_jacobian(forward,event_ndims=1)
    difference = forward_jacobian+inverse_jacobian
    print(np.allclose(forward_jacobian,-inverse_jacobian))
    print((difference/forward_jacobian).numpy()[np.logical_not(np.isclose(forward_jacobian,-inverse_jacobian))])
    print(len((difference/forward_jacobian).numpy()[np.logical_not(np.isclose(forward_jacobian,-inverse_jacobian))])/nevents)
