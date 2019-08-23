import tensorflow as tf
import tensorflow_probability as tfp
from .. import splines
tfb = tfp.bijectors

class CouplingBijector(tfb.Bijector):
    def __init__(self, mask, transform_net_create_fn, blob=None, **kwargs):
        mask = tf.convert_to_tensor(mask)

        super(CouplingBijector,self).__init__(forward_min_event_ndims=1,**kwargs)
        self.features = mask.shape[0]
        features_vector = tf.range(self.features)

        self.identity_features = features_vector[mask <= 0]
        self.transform_features = features_vector[mask > 0]

        assert self.num_identity_features + self.num_transform_features == self.features

        self.blob = False if blob is None else True
        if self.blob:
            if not isinstance(blob,int):
                raise ValueError('Blob encoding requires a number of bins')
            self.nbins_in = int(blob)

        if self.blob:
            self.transform_net = transform_net_create_fn(
                self.num_identity_features*self.nbins_in,
                self.num_transform_features * self._transform_dim_multiplier()
        )
        else:
            self.transform_net = transform_net_create_fn(
                self.num_identity_features,
                self.num_transform_features * self._transform_dim_multiplier()
        )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def _one_blob(self, xd):
        y = tf.tile(((0.5/self.nbins_in) + tf.range(0.,1.,delta = 1./self.nbins_in)),[tf.size(xd)]) 
        y = tf.cast(tf.reshape(y,(-1,self.num_identity_features,self.nbins_in)), dtype=tf.float64)
        res = tf.exp(((-self.nbins_in*self.nbins_in)/2.)*(y-xd[...,tf.newaxis])**2)
        res = tf.reshape(res,(-1,self.num_identity_features*self.nbins_in))
        return res
    
    def _forward(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        if self.blob:
            identity_split_blob = self._one_blob(identity_split)
            transform_params = self.transform_net(identity_split_blob, context)
        else:
            transform_params = self.transform_net(identity_split, context)

        transform_split, logabsdet = self._coupling_transform_forward(
                inputs=transform_split,
                transform_params=transform_params
        )

        outputs = tf.concat([identity_split,transform_split],axis=1)
        indices = tf.concat([self.identity_features, self.transform_features], axis=-1)
        outputs = tf.gather(outputs,indices,axis=1)

        return outputs

    def _inverse(self, inputs, context=None):
        
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        if self.blob:
            identity_split_blob = self._one_blob(identity_split)
            transform_params = self.transform_net(identity_split_blob, context)
        else:
            transform_params = self.transform_net(identity_split, context)
        
        transform_split, logabsdet = self._coupling_transform_inverse(
                inputs=transform_split,
                transform_params=transform_params
        )

        outputs = tf.concat([identity_split,transform_split],axis=1)
        indices = tf.concat([self.identity_features, self.transform_features], axis=-1)
        outputs = tf.gather(outputs,indices,axis=1)

        return outputs

    def _forward_log_det_jacobian(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        if self.blob:
            identity_split_blob = self._one_blob(identity_split)
            transform_params = self.transform_net(identity_split_blob, context)
        else:
            transform_params = self.transform_net(identity_split, context)
        
        transform_split, logabsdet = self._coupling_transform_forward(
                inputs=transform_split,
                transform_params=transform_params
        )

        return logabsdet

    def _inverse_log_det_jacobian(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        if self.blob:
            identity_split_blob = self._one_blob(identity_split)
            transform_params = self.transform_net(identity_split_blob, context)
        else:
            transform_params = self.transform_net(identity_split, context)
        
        transform_split, logabsdet = self._coupling_transform_inverse(
                inputs=transform_split,
                transform_params=transform_params
        )

        return logabsdet

    def _transform_dim_multiplier(self):
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        raise NotImplementedError()

class AffineBijector(CouplingBijector):
    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features:]
        shift = transform_params[:, :self.num_transform_features]
        scale = tf.nn.sigmoid(unconstrained_scale + 2) + 1e-3
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = tf.math.log(scale)
        outputs = inputs * scale + shift
        logabsdet = tf.reduce_sum(log_scale, axis=-1)
        return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = tf.math.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -tf.reduce_sum(log_scale, axis=-1)
        return outputs, logabsdet

class AdditiveBijector(AffineBijector):
    def _transform_dim_multiplier(self):
        return 1

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = tf.ones_like(shift)
        return scale, shift

class PiecewiseBijector(CouplingBijector):
    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        b, d = inputs.shape
        transform_params = tf.reshape(transform_params, (b, d, -1))

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, tf.reduce_sum(logabsdet,axis=-1)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()

class PiecewiseLinear(PiecewiseBijector):
    def __init__(self, mask, transform_net_create_fn, num_bins=10,**kwargs):
        self.num_bins = num_bins
        super(PiecewiseLinear,self).__init__(mask, transform_net_create_fn, **kwargs)

    def _transform_dim_multiplier(self):
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_pdf = transform_params

        return splines.linear_spline(
                inputs=inputs,
                unnormalized_pdf=unnormalized_pdf,
                inverse=inverse
        )

class PiecewiseQuadratic(PiecewiseBijector):
    def __init__(self, mask, transform_net_create_fn, num_bins=10,
            min_bin_width=splines.quadratic.DEFAULT_MIN_BIN_WIDTH,
            min_bin_height=splines.quadratic.DEFAULT_MIN_BIN_HEIGHT,
            **kwargs):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        super(PiecewiseQuadratic,self).__init__(mask, transform_net_create_fn, **kwargs)

    def _transform_dim_multiplier(self):
        return self.num_bins * 2 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:]

        return splines.quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                inverse=inverse,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height
        )

class PiecewiseRationalQuadratic(PiecewiseBijector):
    def __init__(self, mask, transform_net_create_fn, num_bins=10,
            min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
            min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
            min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
            **kwargs):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        super(PiecewiseRationalQuadratic,self).__init__(mask, transform_net_create_fn, **kwargs)

    def _transform_dim_multiplier(self):
        return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., :self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins:2*self.num_bins]
        unnormalized_derivatives = transform_params[..., 2*self.num_bins:]

        return splines.rational_quadratic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnormalized_derivatives=unnormalized_derivatives,
                inverse=inverse,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative
        )


if __name__ == '__main__':
    import numpy as np

    def build_dense(in_features, out_features):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(in_features))
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dense(out_features))
        model.summary()
        return model

    layer = PiecewiseRationalQuadratic([1,1,0,0], build_dense)

    inputs = np.array(np.random.random((10,4)),dtype=np.float32)

    outputs = layer.forward(inputs)
    inputs_inv = layer.inverse(outputs)

    print(inputs[np.logical_not(np.isclose(inputs,inputs_inv))])
    print(inputs_inv[np.logical_not(np.isclose(inputs,inputs_inv))])

    jac_f = layer._forward_log_det_jacobian(inputs)
    jac_b = layer._inverse_log_det_jacobian(outputs)

    print(jac_f[np.logical_not(np.isclose(jac_f,jac_b))])
    print(jac_b[np.logical_not(np.isclose(jac_f,jac_b))])

    bijectors = [layer]
    bijectors.append(PiecewiseRationalQuadratic([0,1,1,0], build_dense))

    bijector = tfb.Chain(bijectors)

    print(inputs[np.logical_not(np.isclose(inputs,bijector.inverse(bijector.forward(inputs))))])
    print(inputs[:10], bijector.forward(inputs[:10]))
