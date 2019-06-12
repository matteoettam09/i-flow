import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
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
        h1 = layers.Dense(16,activation='relu')(inval)
        h2 = layers.Dense(16,activation='relu')(h1)
        out = layers.Dense(d*nbins)(h2)
        out = layers.Reshape((d,nbins))(out)
        model = models.Model(inval,out)
        model.summary()
        return model

    def Q(self, xd):
        QMat = tf.nn.softmax(self.QMat(xd),axis=-1)
        return QMat

    def pdf(self,x):
        xd, xD = x[..., :self.d], x[..., self.d:]
        Q = self.Q(xd)
        ibins = tf.cast(tf.floor(xD*self.nbins)+1,dtype=tf.int32)
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        return tf.concat([tf.ones_like(xd), tf.reduce_sum(Q*one_hot,axis=-1)/self.width], axis=-1)

    def _forward(self, x):
        "Calculate forward coupling layer"
        xd, xD = x[..., :self.d], x[..., self.d:]
        Q = self.Q(xd)
        ibins = tf.cast(tf.floor(xD*self.nbins),dtype=tf.int32)
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        one_hot2 = tf.one_hot(ibins-1,depth=self.nbins)
        yD = (xD*self.nbins-tf.cast(ibins,dtype=tf.float32)) \
           * tf.reduce_sum(Q*one_hot,axis=-1) \
           + tf.reduce_sum(tf.cumsum(Q,axis=-1)*one_hot2,axis=-1)
        return tf.concat([xd, yD], axis=-1)

    def _inverse(self, y):
        "Calculate inverse coupling layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        Q = self.Q(yd)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q,axis=-1),yD[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.reshape(ibins,[tf.shape(yD)[0],self.d])
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        one_hot2 = tf.one_hot(ibins-1,depth=self.nbins)
        xD = (yD-tf.reduce_sum(tf.cumsum(Q,axis=-1)*one_hot2,axis=-1)) \
                *tf.reciprocal(tf.reduce_sum(Q*one_hot,axis=-1)) \
                +tf.cast(ibins,dtype=tf.float32)*self.width
        return tf.concat([yd, xD], axis=-1)

    def _forward_log_det_jacobian(self, x):
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[...,self.d:]),axis=-1)

    def _inverse_log_det_jacobian(self, y):
        "Calculate log determinant of Coupling Layer"
        yd, yD = y[..., :self.d], y[..., self.d:]
        Q = self.Q(yd)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q,axis=-1),yD[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.reshape(ibins,[tf.shape(yD)[0],self.d])
        one_hot = tf.one_hot(ibins,depth=self.nbins)
        return -tf.reduce_sum(tf.log(tf.reduce_sum(Q*one_hot,axis=-1)/self.width),axis=-1)

class PiecewiseQuadratic(tfb.Bijector):
    """
    Piecewise Quadratic: based on 1808.03856
    """
    def __init__(self, D, d, nbins, layer_id=0, validate_args=False, name="PiecewiseQuadratic"):
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
        self.range = tf.range(self.d)
#        self.WMat = self.buildW(self.d, self.nbins)
#        self.VMat = self.buildV(self.d, self.nbins)
        self.NNMat = self.buildNN(self.d, self.nbins)
        self.trainable_variables = [
                self.NNMat.trainable_variables,
#                self.VMat.trainable_variables,
        ]

#    def buildW(self, d, nbins):
#        inval = layers.Input(shape=(d,))
#        h1 = layers.Dense(16,activation='relu')(inval)
#        h2 = layers.Dense(16,activation='relu')(h1)
#        out = layers.Dense(d*nbins,activation='relu')(h2)
#        out = layers.Reshape((d,nbins))(out)
#        model = models.Model(inval,out)
#        return model
#
#    def buildV(self, d, nbins):
#        inval = layers.Input(shape=(d,))
#        h1 = layers.Dense(16,activation='relu')(inval)
#        h2 = layers.Dense(16,activation='relu')(h1)
#        out = layers.Dense(d*(nbins+1),activation='relu')(h2)
#        out = layers.Reshape((d,nbins+1))(out)
#        model = models.Model(inval,out)
#        return model
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

    def buildNN(self, d, nbins):
        inval = layers.Input(shape=(d,))
        h1 = layers.Dense(64,activation='relu')(inval)
        h2 = layers.Dense(64,activation='relu')(h1)
        out = layers.Dense(d*(2*nbins+1),activation='relu')(h2)
        out = layers.Reshape((d,2*nbins+1))(out)
        model = models.Model(inval,out)
        model.summary()
        return model

    def GetWV(self, xd):
        NNMat = self.NNMat(xd)
        W = tf.nn.softmax(NNMat[...,:self.nbins],axis=-1)
        V = NNMat[...,self.nbins:]
        VExp = tf.exp(V)
        VSum = tf.reduce_sum((VExp[...,:self.nbins]+VExp[...,1:])*W/2,axis=-1,keepdims=True)
        V = tf.truediv(VExp,VSum)
        return W, V

    def _find_bins(self,x,y):
        ibins = tf.cast(tf.searchsorted(y,x[...,tf.newaxis],side='right'),dtype=tf.int32)
        ibins = tf.reshape(ibins,[tf.shape(x)[0],self.d])
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

    def _forward(self, x):
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

    def _inverse(self, y):
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
                tf.div_no_nan(1.,denom)*(-Vbins+tf.sqrt(Vbins**2+2*beta*denom))
        )
        xD = tf.reduce_sum(W*one_hot,axis=-1)*xD + tf.reduce_sum(WSum*one_hot_sum,axis=-1)
#        xD = tf.where(tf.is_nan(xD), tf.ones_like(xD), xD)
        return tf.concat([yd, xD], axis=-1)

    def _forward_log_det_jacobian(self, x):
        "Calculate log determinant of Coupling Layer"
        return tf.reduce_sum(tf.log(self.pdf(x)[...,self.d:]),axis=-1)

    def _inverse_log_det_jacobian(self, y):
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

