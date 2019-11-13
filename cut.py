""" Test cut efficiency """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from flow.integration import integrator
from flow.integration import couplings
from flow.splines.spline import _knot_positions, _search_sorted
from flow.splines.spline import _gather_squeeze

tfd = tfp.distributions  # pylint: disable=invalid-name
tf.keras.backend.set_floatx('float64')


def func(pts_x):
    """ Calculate function for testing. """
    alpha = 1.0
    cut_value = 0.05
    return tf.where(pts_x[:, 0] > cut_value, tf.pow(pts_x[:, 0], -alpha), 0)


def get_spline(inputs, widths, heights, derivatives):
    """ Get the points of the splines to plot. """
    min_bin_width = 1e-15
    min_bin_height = 1e-15
    min_derivative = 1e-15

    num_bins = widths.shape[-1]

    widths = tf.nn.softmax(widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = _knot_positions(widths, 0)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = ((min_derivative + tf.nn.softplus(derivatives))
                   / (tf.cast(min_derivative + tf.math.log(2.), tf.float64)))

    heights = tf.nn.softmax(heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = _knot_positions(heights, 0)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = _search_sorted(cumwidths, inputs)

    input_cumwidths = _gather_squeeze(cumwidths, bin_idx)
    input_bin_widths = _gather_squeeze(widths, bin_idx)

    input_cumheights = _gather_squeeze(cumheights, bin_idx)
    delta = heights / widths
    input_delta = _gather_squeeze(delta, bin_idx)

    input_derivatives = _gather_squeeze(derivatives, bin_idx)
    input_derivatives_p1 = _gather_squeeze(derivatives[..., 1:], bin_idx)

    input_heights = _gather_squeeze(heights, bin_idx)

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (input_delta * theta**2
                                 + input_derivatives
                                 * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_p1
                                  - 2 * input_delta)
                                 * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator

    return outputs, cumwidths, cumheights


def plot_spline(widths, heights, derivatives):
    """ Plot the spline. """
    nsamples = 10000
    # nodes = 5

    pts_x = np.linspace(0, 1, nsamples).reshape(nsamples, 1)
    widths = np.array([widths.numpy().tolist()]*nsamples)
    heights = np.array([heights.numpy().tolist()]*nsamples)
    derivatives = np.array([derivatives.numpy().tolist()]*nsamples)

    outputs, widths, heights = get_spline(pts_x, widths, heights, derivatives)

    plt.plot(pts_x, outputs, zorder=1)
    plt.scatter(widths.numpy(), heights.numpy(), s=20, color='red', zorder=2)


def build(in_features, out_features, options):
    " Build the NN. """
    del options

    invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
    hidden = tf.keras.layers.Dense(128, activation='relu')(invals)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_features, activation='relu')(hidden)
    model = tf.keras.models.Model(invals, outputs)
    model.summary()
    return model


def main():
    """ Main function """
    bijector = couplings.PiecewiseRationalQuadratic([1, 0], build,
                                                    num_bins=10,
                                                    blob=None,
                                                    options=None)

    low = np.array([0, 0], dtype=np.float64)
    high = np.array([1, 1], dtype=np.float64)
    dist = tfd.Uniform(low=low, high=high)
    dist = tfd.Independent(distribution=dist,
                           reinterpreted_batch_ndims=1)
    dist = tfd.TransformedDistribution(
        distribution=dist,
        bijector=bijector)

    optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=10.0)
    integrate = integrator.Integrator(func, dist, optimizer,
                                      loss_func='exponential')

    for i in range(10):
        point = float(i)/10.0
        transform_params = bijector.transform_net(np.array([[point]]))
        widths = transform_params[..., :10]
        heights = transform_params[..., 10:20]
        derivatives = transform_params[..., 20:]
        plot_spline(widths, heights, derivatives)

    plt.savefig('pretraining.png')
    plt.show()

    for epoch in range(500):
        loss, integral, error = integrate.train_one_step(10000,
                                                         integral=True)
        if epoch % 10 == 0:
            print('Epoch: {:3d} Loss = {:8e} Integral = '
                  '{:8e} +/- {:8e}'.format(epoch, loss, integral, error))

    for i in range(10):
        point = float(i)/10.0
        transform_params = bijector.transform_net(np.array([[point]]))
        widths = transform_params[..., :10]
        heights = transform_params[..., 10:20]
        derivatives = transform_params[..., 20:]
        plot_spline(widths, heights, derivatives)

    plt.savefig('posttraining.png')
    plt.show()


if __name__ == '__main__':
    main()
