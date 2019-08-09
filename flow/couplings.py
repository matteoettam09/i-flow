import tensorflow as tf
import tensorflow_probability as tfp
import splines
tfb = tfp.bijectors

class CouplingBijector(tfb.Bijector):
    def __init__(self, mask, transform_net_create_fn, **kwargs):
        mask = tf.convert_to_tensor(mask)

        super().__init__(forward_min_event_ndims=1,**kwargs)
        self.features = mask.shape[0]
        features_vector = tf.range(self.features)

        self.identity_features = features_vector[mask <= 0]
        self.transform_features = features_vector[mask > 0]

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

    def _forward(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
                inputs=transform_split,
                transform_params=transform_params
        )

        outputs = tf.concat([identity_split,transform_split],axis=1)
        indices = tf.concat([self.identity_features, self.transform_features], axis=-1)
        indices = self.features - indices - 1
        outputs = tf.gather(outputs,indices,axis=1)

        return outputs

    def _inverse(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_inverse(
                inputs=transform_split,
                transform_params=transform_params
        )

        outputs = tf.concat([identity_split,transform_split],axis=1)
        indices = tf.concat([self.identity_features, self.transform_features], axis=-1)
        indices = self.features - indices - 1
        outputs = tf.gather(outputs,indices,axis=1)

        return outputs

    def _forward_log_det_jacobian(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
                inputs=transform_split,
                transform_params=transform_params
        )

        return logabsdet

    def _inverse_log_det_jacobian(self, inputs, context=None):
        identity_split = tf.gather(inputs,self.identity_features,axis=-1)
        transform_split = tf.gather(inputs,self.transform_features,axis=-1)

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
        super().__init__(mask, transform_net_create_fn, **kwargs)

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

        super().__init__(mask, transform_net_create_fn, **kwargs)

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

        super().__init__(mask, transform_net_create_fn, **kwargs)

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

    layer = PiecewiseRationalQuadratic([1,0,0,1],build_dense)

    inputs = np.array(np.random.random((100000,4)),dtype=np.float32)

    outputs = layer.forward(inputs)
    inputs_inv = layer.inverse(outputs)

    print(inputs[np.logical_not(np.isclose(inputs,inputs_inv))])
    print(inputs_inv[np.logical_not(np.isclose(inputs,inputs_inv))])

    print(layer._forward_log_det_jacobian(inputs))


