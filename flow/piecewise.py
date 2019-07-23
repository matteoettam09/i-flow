# -*- coding: utf-8 -*-
"""Implement the piecewise coupling layers.

This module implements the piecewise base coupling layer, the piecewise linear coupling layer,
the piecewise quadratic coupling layer, and the piecewise quadratic with const bin width
coupling layer. 

Todo:
    * Debug piecewise quadratic layers
"""

from tensorflow.keras import layers, models
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors


class Piecewise(tfb.Bijector):

    """Implement base class for piecewise coupling layers."""

    def __init__(self, D, d, nbins, nchannels=1, blob=False, hot=False, model=None, unet=False, **kwargs):
        """Initialize the global piecewise variables.

        Args:
            D (int): Total number of dimensions.
            d (int): Number of dimensions to train on.
            nbins (int): Number of bins for the calculation.
            blob (bool): Flag for one blob encoding.
            hot (bool): Flag for one hot encoding.
            model (tf.keras.models): User defined model to use for the neural network.
            unet (bool): Flag to use a default unet network.
            **kwargs: Arguments to be passed to tfb.Bijector.
        """
        super(Piecewise, self).__init__(**kwargs)
        self.D, self.d = D, d
        self.nbins = nbins
        self.nchannels = nchannels
        self.range = tf.cast(tf.range(self.d), dtype=tf.int32)
        self.blob = blob
        self.hot = hot
        self.model = model
        self.unet = unet
        self.width = 1.0/self.nbins

    def _one_blob(self, xd):
        y = tf.tile(((0.5*self.width) + tf.range(0., 1.,
                                                 delta=1./self.nbins)), [tf.size(input=xd)])
        y = tf.reshape(y, (-1, self.d, self.nbins))
        res = tf.exp(((-self.nbins*self.nbins)/2.)*(y-xd[..., tf.newaxis])**2)
        return res

    def _one_hot(self, xd):
        ibins = tf.cast(tf.floor(xd*self.nbins), dtype=tf.int32)
        ibins = tf.compat.v1.where(tf.equal(ibins, self.nbins *
                                  tf.ones_like(ibins)), ibins-1, ibins)
        one_hot = tf.one_hot(ibins, depth=self.nbins, axis=-1)
        one_hot = tf.reshape(one_hot, [-1, self.d*self.nbins])
        return one_hot

    def _build_input(self):
        shape = self.d

        if self.blob or self.hot:
            shape *= self.nbins

        if self.nchannels != 1:
            shape += self.nchannels

        return layers.Input(shape=(shape,))

    def _build_unet(self, inval):
        h1 = layers.Dense(256, activation='relu')(inval)
        h2 = layers.Dense(128, activation='relu')(h1)
        h3 = layers.Dense(64, activation='relu')(h2)
        h4 = layers.Dense(32, activation='relu')(h3)
        h5 = layers.Dense(32, activation='relu')(h4)
        h5 = layers.concatenate([h5, h3], axis=-1)
        h6 = layers.Dense(64, activation='relu')(h5)
        h6 = layers.concatenate([h6, h2], axis=-1)
        h7 = layers.Dense(128, activation='relu')(h6)
        h7 = layers.concatenate([h7, h1], axis=-1)
        h8 = layers.Dense(256, activation='relu')(h7)

        return h8

    def _build_dense(self, inval):
        h = layers.Dense(128, activation='relu')(inval)
        for i in range(4):
            h = layers.Dense(128, activation='relu')(h)

        return h

    def _build_model(self, nbins):
        inval = self._build_input()

        if self.model is not None:
            if isinstance(self.model, models.Model):
                model = self.model
                model.summary()
                return model
            else:
                print('WARNING: Passed in network is not valid. Defaulting to Unet.')
                unet = self._build_unet(inval)
                out = layers.Dense((self.D-self.d)*nbins)(unet)
        elif self.unet:
            unet = self._build_unet(inval)
            out = layers.Dense((self.D-self.d)*nbins)(unet)
        else:
            dense = self._build_dense(inval)
            out = layers.Dense((self.D-self.d)*nbins)(dense)

        out = layers.Reshape(((self.D-self.d), nbins))(out)
        model = models.Model(inval, out)
        model.summary()
        return model

    def _channel_encode(self,xd,channel):
        channel_hot = tf.one_hot(tf.cast(channel,dtype=tf.int32),depth=self.nchannels, axis=-1)
        return tf.concat([xd,channel_hot],axis=-1)


class PiecewiseLinear(Piecewise):

    """Implement PiecewiseLinear coupling layer.

    Piecewise Linear is based on arXiv:1808.03856. It uses piecewise linear 
    CDFs to estimate the true CDF.

    """

    def __init__(self, D, d, nbins, layer_id=0, validate_args=False,
                 name="PiecewiseLinear", **kwargs):
        """Initialize the Piecewise Linear coupling layer.

        Args:
            D (int): Number of dimensions.
            d (int): First d units are pass-thru units.
            nbins (int): Number of bins for the network.
            layer_id (int): ID to label the specfic layer.
            validate_args (bool): To be passed to base class.
            name (str): Name of the layer to be passed to the base class.
            **kwargs: Additional arguments to be passed to the base class.
        """
        super(PiecewiseLinear, self).__init__(D, d, nbins,
                                              forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs
                                              )
        self.id = layer_id
        self.QMat = self._build_model(self.nbins)
        self.trainable_variables = self.QMat.trainable_variables

    def _q(self, xd, channel):
        if self.hot:
            xd = self._one_hot(xd)
        elif self.blob:
            xd = tf.reshape(self._one_blob(xd), [-1, self.d*self.nbins])

        if self.nchannels != 1:
            xd = self._channel_encode(xd, channel)

        return tf.nn.softmax(self.QMat(xd), axis=-1)

    def _pdf(self, x, channel):
        xd, xD = x[..., :self.d], x[..., self.d:]
        Q = self._q(xd, channel)
        ibins = tf.cast(tf.floor(xD*self.nbins), dtype=tf.int32)
        ibins = tf.compat.v1.where(tf.equal(ibins, self.nbins *
                                  tf.ones_like(ibins)), ibins-1, ibins)
        one_hot = tf.one_hot(ibins, depth=self.nbins)
        return tf.concat([tf.ones_like(xd), tf.reduce_sum(input_tensor=Q*one_hot, axis=-1)/self.width], axis=-1)

    def _inverse(self, x, channel = 1):
        xd, xD = x[..., :self.d], x[..., self.d:]
        Q = self._q(xd, channel)
        ibins = tf.cast(tf.floor(xD*self.nbins), dtype=tf.int32)
        ibins = tf.compat.v1.where(tf.equal(ibins, self.nbins *
                                  tf.ones_like(ibins)), ibins-1, ibins)
        one_hot = tf.one_hot(ibins, depth=self.nbins)
        one_hot2 = tf.one_hot(ibins-1, depth=self.nbins)
        yD = ((xD*self.nbins-tf.cast(ibins, dtype=tf.float32))
              * tf.reduce_sum(input_tensor=Q*one_hot, axis=-1)) \
            + tf.reduce_sum(input_tensor=tf.cumsum(Q, axis=-1)*one_hot2, axis=-1)
        return tf.concat([xd, yD], axis=-1)

    def _forward(self, y, channel = 1):
        yd, yD = y[..., :self.d], y[..., self.d:]
        Q = self._q(yd, channel)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q, axis=-1),
                                        yD[..., tf.newaxis], side='right'), dtype=tf.int32)
        ibins = tf.compat.v1.where(tf.equal(ibins, self.nbins *
                                  tf.ones_like(ibins)), ibins-1, ibins)
        ibins = tf.reshape(ibins, [tf.shape(input=yD)[0], self.D-self.d])
        one_hot = tf.one_hot(ibins, depth=self.nbins)
        one_hot2 = tf.one_hot(ibins-1, depth=self.nbins)
        xD = ((yD-tf.reduce_sum(input_tensor=tf.cumsum(Q, axis=-1)*one_hot2, axis=-1))
              * tf.math.reciprocal(tf.reduce_sum(input_tensor=Q*one_hot, axis=-1))
              + tf.cast(ibins, dtype=tf.float32))*self.width
        return tf.concat([yd, xD], axis=-1)

    def _inverse_log_det_jacobian(self, x, channel = 1):
        return tf.reduce_sum(input_tensor=tf.math.log(self._pdf(x, channel)[..., self.d:]), axis=-1)

    def _forward_log_det_jacobian(self, y, channel = 1):
        yd, yD = y[..., :self.d], y[..., self.d:]
        Q = self._q(yd, channel)
        ibins = tf.cast(tf.searchsorted(tf.cumsum(Q, axis=-1),
                                        yD[..., tf.newaxis], side='right'), dtype=tf.int32)
        ibins = tf.compat.v1.where(tf.equal(ibins, self.nbins *
                                  tf.ones_like(ibins)), ibins-1, ibins)
        ibins = tf.reshape(ibins, [tf.shape(input=yD)[0], self.D-self.d])
        one_hot = tf.one_hot(ibins, depth=self.nbins)
        return -tf.reduce_sum(input_tensor=tf.math.log(tf.reduce_sum(input_tensor=Q*one_hot, axis=-1)/self.width), axis=-1)


class PiecewiseQuadratic(Piecewise):

    """Implement PiecewiseQuadratic coupling layer.

    Piecewise Quadratic is based on arXiv:1808.03856. It uses piecewise quadratic 
    CDFs to estimate the true CDF. In this implementation,
    the bin widths are allowed to float to better describe the function.

    """

    def __init__(self, D, d, nbins, layer_id=0, validate_args=False,
                 name="PiecewiseQuadratic", **kwargs):
        """Initialize the Piecewise Quadratic coupling layer.

        Args:
            D (int): Number of dimensions.
            d (int): First d units are pass-thru units.
            nbins (int): Number of bins for the network.
            layer_id (int): ID to label the specfic layer.
            validate_args (bool): To be passed to base class.
            name (str): Name of the layer to be passed to the base class.
            **kwargs: Additional arguments to be passed to the base class.
        """
        super(PiecewiseQuadratic, self).__init__(D, d, nbins,
                                                 forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs
                                                 )

        self.id = layer_id
        self.NNMat = self._build_model(2*self.nbins+1)
        self.trainable_variables = self.NNMat.trainable_variables

    def _get_wv(self, xd, channel):
        if self.hot:
            xd = self._one_hot(xd)
        elif self.blob:
            xd = tf.reshape(self._one_blob(xd), [-1, self.d*self.nbins])

        if self.nchannels != 1:
            xd = self._channel_encode(xd, channel)

        NNMat = self.NNMat(xd)
        W = tf.nn.softmax(NNMat[..., :self.nbins], axis=-1)
        W = tf.compat.v1.where(tf.less(W, 1e-6*tf.ones_like(W)), 1e-6*tf.ones_like(W), W)
        V = NNMat[..., self.nbins:]
        VExp = tf.exp(V)
        VSum = tf.reduce_sum(
            input_tensor=(VExp[..., :self.nbins]+VExp[..., 1:])*W/2, axis=-1, keepdims=True)
        V = tf.truediv(VExp, VSum)
        return W, V

    def _find_bins(self, x, y):
        ibins = tf.cast(tf.searchsorted(
            y, x[..., tf.newaxis], side='right'), dtype=tf.int32)
        ibins = tf.compat.v1.where(tf.equal(ibins, self.nbins *
                                  tf.ones_like(ibins)), ibins-1, ibins)
        ibins = tf.reshape(ibins, [tf.shape(input=x)[0], self.D-self.d])
        one_hot = tf.one_hot(ibins, depth=self.nbins)
        one_hot_sum = tf.one_hot(ibins-1, depth=self.nbins)
        one_hot_V = tf.one_hot(ibins, depth=self.nbins+1)

        return one_hot, one_hot_sum, one_hot_V

    def _pdf(self, x, channel):
        xd, xD = x[..., :self.d], x[..., self.d:]
        W, V = self._get_wv(xd, channel)
        WSum = tf.cumsum(W, axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD, WSum)
        alpha = (xD-tf.reduce_sum(input_tensor=WSum*one_hot_sum, axis=-1)) \
            * tf.math.reciprocal(tf.reduce_sum(input_tensor=W*one_hot, axis=-1))
        result = tf.reduce_sum(input_tensor=(V[..., 1:]-V[..., :-1])*one_hot, axis=-1)*alpha \
            + tf.reduce_sum(input_tensor=V*one_hot_V, axis=-1)
        return tf.concat([xd, result], axis=-1)

    def _inverse(self, x, channel = 1):
        xd, xD = x[..., :self.d], x[..., self.d:]
        W, V = self._get_wv(xd, channel)
        WSum = tf.cumsum(W, axis=-1)
        VSum = tf.cumsum((V[..., 1:]+V[..., :-1])*W/2.0, axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(xD, WSum)
        alpha = (xD-tf.reduce_sum(input_tensor=WSum*one_hot_sum, axis=-1)) \
            * tf.math.reciprocal(tf.reduce_sum(input_tensor=W*one_hot, axis=-1))
        yD = alpha**2/2*tf.reduce_sum(input_tensor=(V[..., 1:]-V[..., 0:-1])*one_hot, axis=-1) \
            * tf.reduce_sum(input_tensor=W*one_hot, axis=-1) \
            + alpha*tf.reduce_sum(input_tensor=V*one_hot_V, axis=-1)*tf.reduce_sum(input_tensor=W*one_hot, axis=-1) \
            + tf.reduce_sum(input_tensor=VSum*one_hot_sum, axis=-1)
        return tf.concat([xd, yD], axis=-1)

    def _forward(self, y, channel = 1):
        yd, yD = y[..., :self.d], y[..., self.d:]
        W, V = self._get_wv(yd, channel)
        WSum = tf.cumsum(W, axis=-1)
        VSum = tf.cumsum((V[..., 1:]+V[..., 0:-1])*W/2.0, axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD, VSum)
        denom = tf.reduce_sum(input_tensor=(V[..., 1:]-V[..., 0:-1])*one_hot, axis=-1)
        beta = (yD - tf.reduce_sum(input_tensor=VSum*one_hot_sum, axis=-1)) \
            * tf.math.reciprocal(tf.reduce_sum(input_tensor=W*one_hot, axis=-1))
        Vbins = tf.reduce_sum(input_tensor=V*one_hot_V, axis=-1)
        xD = tf.compat.v1.where(tf.equal(tf.zeros_like(denom), denom),
                      beta/Vbins,
                      tf.math.divide_no_nan(
                          (-Vbins+tf.sqrt(Vbins**2+2*beta*denom)), denom)
                      )
        xD = tf.reduce_sum(input_tensor=W*one_hot, axis=-1)*xD + \
            tf.reduce_sum(input_tensor=WSum*one_hot_sum, axis=-1)
        return tf.concat([yd, xD], axis=-1)

    def _inverse_log_det_jacobian(self, x, channel = 1):
        return tf.reduce_sum(input_tensor=tf.math.log(self._pdf(x, channel)[..., self.d:]), axis=-1)

    def _forward_log_det_jacobian(self, y, channel = 1):
        yd, yD = y[..., :self.d], y[..., self.d:]
        W, V = self._get_wv(yd, channel)
        WSum = tf.cumsum(W, axis=1)
        VSum = tf.cumsum((V[..., 1:]+V[..., 0:-1])*W/2.0, axis=-1)
        one_hot, one_hot_sum, one_hot_V = self._find_bins(yD, VSum)
        denom = tf.reduce_sum(input_tensor=(V[..., 1:]-V[..., 0:-1])*one_hot, axis=-1)
        beta = (yD - tf.reduce_sum(input_tensor=VSum*one_hot_sum, axis=-1)) \
            * tf.math.reciprocal(tf.reduce_sum(input_tensor=W*one_hot, axis=-1))
        Vbins = tf.reduce_sum(input_tensor=V*one_hot_V, axis=-1)
        alpha = tf.compat.v1.where(tf.equal(tf.zeros_like(denom), denom),
                         beta/Vbins,
                         tf.math.divide_no_nan(
                             (-Vbins+tf.sqrt(Vbins**2+2*beta*denom)), denom)
                         )
        result = tf.reduce_sum(
            input_tensor=(V[..., 1:]-V[..., 0:-1])*one_hot, axis=-1)*alpha+Vbins
        return -tf.reduce_sum(input_tensor=tf.math.log(result), axis=-1)


class PiecewiseQuadraticConst(PiecewiseQuadratic):

    """Implement PiecewiseQuadraticConst coupling layer.

    Piecewise Quadratic with constant bin widths is based on arXiv:1808.03856.
    It uses piecewise quadratic CDFs to estimate the true CDF. In this implementation,
    the bin widths are fixed to constant values similar to the piecewise linear
    coupling layers.

    """

    def __init__(self, D, d, nbins, layer_id=0, validate_args=False,
                 name="PiecewiseQuadraticConst", **kwargs):
        """Initialize the Piecewise Quadratic Constant coupling layer.

        Args:
            D (int): Number of dimensions.
            d (int): First d units are pass-thru units.
            nbins (int): Number of bins for the network.
            layer_id (int): ID to label the specfic layer.
            validate_args (bool): To be passed to base class.
            name (str): Name of the layer to be passed to the base class.
            **kwargs: Additional arguments to be passed to the base class.
        """
        super(PiecewiseQuadratic, self).__init__(D, d, nbins,
                                                 forward_min_event_ndims=1, validate_args=validate_args, name=name, **kwargs
                                                 )

        self.id = layer_id
        self.VMat = self._build_model(self.nbins+1)
        self.trainable_variables = self.VMat.trainable_variables

    def _w(self, xd):
        return tf.constant(1./self.nbins, shape=(xd.shape[0], self.D-self.d, self.nbins))

    def _v(self, xd, W, channel):
        if self.hot:
            xd = self._one_hot(xd)
        elif self.blob:
            xd = tf.reshape(self._one_blob(xd), [-1, self.d*self.nbins])

        if self.nchannels != 1:
            xd = self._channel_encode(xd, channel)

        VMat = self.VMat(xd)
        VExp = tf.exp(VMat)
        VSum = tf.reduce_sum(input_tensor=(VExp[..., 0:self.nbins]+VExp[..., 1:self.nbins+1])
                             * W[..., :self.nbins]/2, axis=-1, keepdims=True)
        VMat = tf.truediv(VExp, VSum)
        return VMat

    def _get_wv(self, xd, channel):
        W = self.W(xd)
        return W, self.V(xd, W, channel)
