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
        self.range = tf.cast(tf.range(self.d),dtype=tf.int32)
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
        QMat = tf.pad(QMat,[[0,0],[0,0],[1,0]])
        return QMat
        
    def pdf(self,x):
        xd, xD = x[:, :self.d], x[:, self.d:]
        Q = self.Q(xd)
        ibins = tf.cast(tf.floor(xD*self.nbins)+1,dtype=tf.int32)
        batch_range = tf.range(tf.shape(xD)[0])
        grid = tf.transpose(tf.meshgrid(batch_range,self.range))
        bins = tf.concat([grid,tf.gather_nd(ibins,grid)[...,tf.newaxis]],axis=2)
        return tf.concat([xd, tf.gather_nd(Q,bins)/self.width], axis=1)
        
    def _forward(self, x):
        "Calculate forward coupling layer"
        xd, xD = x[:, :self.d], x[:, self.d:]
        Q = self.Q(xd)
        ibins = tf.cast(tf.floor(xD*self.nbins),dtype=tf.int32)
        ibinsp1 = ibins+1
        batch_range = tf.range(tf.shape(xD)[0])
        grid = tf.transpose(tf.meshgrid(batch_range,self.range))
        bins = tf.concat([grid,tf.gather_nd(ibinsp1,grid)[...,tf.newaxis]],axis=2)
        binsSum = tf.concat([grid,tf.gather_nd(ibins,grid)[...,tf.newaxis]],axis=2)
        yD = (xD*self.nbins-tf.cast(ibins,dtype=tf.float32))*tf.gather_nd(Q,bins)+tf.gather_nd(tf.cumsum(Q,axis=2),binsSum)
        return tf.concat([xd, yD], axis=1)
        
    def _inverse(self, y):
        "Calculate inverse coupling layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        Q = self.Q(yd)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q,axis=2),yD[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.reshape(ibins,[tf.shape(yD)[0],self.d])
        ibins = ibins-1
        ibinsp1 = ibins+1
        batch_range = tf.range(tf.shape(yD)[0])
        grid = tf.transpose(tf.meshgrid(batch_range,self.range))
        bins = tf.concat([grid,tf.gather_nd(ibinsp1,grid)[...,tf.newaxis]],axis=2)
        binsSum = tf.concat([grid,tf.gather_nd(ibins,grid)[...,tf.newaxis]],axis=2)
        xD = ((yD-tf.gather_nd(tf.cumsum(Q,axis=2),binsSum))*tf.reciprocal(tf.gather_nd(Q,bins))\
              +tf.cast(ibins,dtype=tf.float32))*self.width
        xD = tf.where(tf.is_nan(xD), tf.ones_like(xD), xD)
        return tf.concat([yd, xD], axis=1)
    
    def _forward_log_det_jacobian(self, x):
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[:,self.d:]),axis=-1)
    
    def _inverse_log_det_jacobian(self, y):
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        Q = self.Q(yd)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q,axis=2),yD[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.reshape(ibins,[tf.shape(yD)[0],self.d])
        ibins = ibins-1
        ibinsp1 = ibins+1
        batch_range = tf.range(tf.shape(yD)[0])
        grid = tf.transpose(tf.meshgrid(batch_range,self.range))
        bins = tf.concat([grid,tf.gather_nd(ibinsp1,grid)[...,tf.newaxis]],axis=2)
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
        self.range = tf.range(self.d)
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
        WMat = tf.pad(WMat,[[0,0],[0,0],[1,0]])
        return WMat
              
    def V(self, xd, W):
        VMat = self.VMat(xd)
        VExp = tf.exp(VMat)
        VSum = tf.reduce_sum((VExp[...,0:self.nbins]+VExp[...,1:self.nbins+1])*W[...,1:self.nbins+1]/2,axis=2,keepdims=True)
        VMat = tf.truediv(VExp,VSum)
        VMat = tf.pad(VMat,[[0,0],[0,0],[1,0]])
        return VMat

    def _find_bins(self,x,y):
        ibins = tf.cast(tf.searchsorted(y,x[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.reshape(ibins,[tf.shape(x)[0],self.d])
        ibins = ibins-1
        ibinsp1 = ibins+1
        batch_range = tf.range(tf.shape(x)[0])
        grid = tf.transpose(tf.meshgrid(batch_range,self.range))
        bins = tf.concat([grid,tf.gather_nd(ibinsp1,grid)[...,tf.newaxis]],axis=2)
        binsSum = tf.concat([grid,tf.gather_nd(ibins,grid)[...,tf.newaxis]],axis=2)
        return bins, binsSum
        
    def pdf(self,x):
        xd, xD = x[:, :self.d], x[:, self.d:]
        W = self.W(xd)
        V = self.V(xd,W)
        WSum = tf.cumsum(W,axis=2)
        bins, binsSum = self._find_bins(xD,WSum)
        alpha = (xD-tf.gather_nd(WSum,binsSum))*tf.reciprocal(tf.gather_nd(W,bins))
        result = tf.gather_nd((V[...,1:]-V[...,0:-1]),bins)*alpha+tf.gather_nd(V,bins)
        return tf.concat([xd, result], axis=1) 
        
    def _forward(self, x):
        "Calculate forward coupling layer"
        xd, xD = x[:, :self.d], x[:, self.d:]
        W = self.W(xd)
        V = self.V(xd,W)
        WSum = tf.cumsum(W,axis=2)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=2)
        bins, binsSum = self._find_bins(xD,WSum)
        alpha = (xD-tf.gather_nd(WSum,binsSum))/tf.gather_nd(W,bins)
        yD = alpha**2/2*tf.gather_nd((V[...,1:]-V[...,0:-1]),bins)*tf.gather_nd(W,bins) \
           + alpha*tf.gather_nd(V,bins)*tf.gather_nd(W,bins) \
           + tf.gather_nd(VSum,binsSum)
        return tf.concat([xd, yD], axis=1)
    
    def _inverse(self, y):
        "Calculate inverse coupling layer"
        yd, yD = y[:, :self.d], y[:, self.d:]
        W = self.W(yd)
        V = self.V(yd,W)
        WSum = tf.cumsum(W,axis=2)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=2)
        bins, binsSum = self._find_bins(yD,VSum)
        denom = tf.gather_nd((V[...,1:]-V[...,0:-1]),bins)
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
        WSum = tf.cumsum(W,axis=1)
        VSum = tf.cumsum((V[...,1:]+V[...,0:-1])*W/2.0,axis=2)
        bins, binsSum = self._find_bins(yD,VSum)
        denom = tf.gather_nd((V[...,1:]-V[...,0:-1]),bins)
        beta = (yD - tf.gather_nd(VSum,binsSum))/tf.gather_nd(W,bins)
        Vbins = tf.gather_nd(V,bins)
        alpha = tf.where(tf.equal(tf.zeros_like(denom),denom),
                      beta/Vbins,
                      1/denom*(-Vbins+tf.sqrt(Vbins**2+2*beta*denom))
                      )
        result = tf.gather_nd((V[...,1:]-V[...,0:-1]),bins)*alpha+Vbins
        return -tf.reduce_sum(tf.log(result),axis=-1)

# Tests if run using main
if __name__ == '__main__':
    nevents = 1000
    NP_DTYPE=np.float32

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
