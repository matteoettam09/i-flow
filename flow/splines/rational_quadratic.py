""" Implement Rational Quadratic splines. """

# pylint: disable=too-many-arguments, too-many-locals, invalid-name

import tensorflow as tf
from .spline import _knot_positions, _gather_squeeze, _search_sorted

DEFAULT_MIN_BIN_WIDTH = 1e-15
DEFAULT_MIN_BIN_HEIGHT = 1e-15
DEFAULT_MIN_DERIVATIVE = 1e-15


def rational_quadratic_spline(inputs,
                              unnormalized_widths,
                              unnormalized_heights,
                              unnormalized_derivatives,
                              inverse=False,
                              left=0., right=1., bottom=0., top=1.,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    """ Definition of rational quadratic spline. """

    out_of_bounds = (inputs < left) | (inputs > right)
    tf.where(out_of_bounds, tf.cast(left, dtype=inputs.dtype), inputs)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = _knot_positions(widths, 0)
    cumwidths = (right - left) * cumwidths + left
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = ((min_derivative + tf.nn.softplus(unnormalized_derivatives))
                   / (tf.cast(min_derivative + tf.math.log(2.), tf.float64)))

    heights = tf.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = _knot_positions(heights, 0)
    cumheights = (top - bottom) * cumheights + bottom
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _search_sorted(cumheights, inputs)
    else:
        bin_idx = _search_sorted(cumwidths, inputs)

    input_cumwidths = _gather_squeeze(cumwidths, bin_idx)
    input_bin_widths = _gather_squeeze(widths, bin_idx)

    input_cumheights = _gather_squeeze(cumheights, bin_idx)
    delta = heights / widths
    input_delta = _gather_squeeze(delta, bin_idx)

    input_derivatives = _gather_squeeze(derivatives, bin_idx)
    input_derivatives_p1 = _gather_squeeze(derivatives[..., 1:], bin_idx)

    input_heights = _gather_squeeze(heights, bin_idx)

    if inverse:
        a = ((inputs - input_cumheights) * (input_derivatives
                                            + input_derivatives_p1
                                            - 2 * input_delta)
             + input_heights * (input_delta - input_derivatives))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_p1
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b**2 - 4 * a * c

        theta = (2 * c) / (-b - tf.sqrt(discriminant))
        outputs = theta * input_bin_widths + input_cumwidths

        theta_one_minus_theta = theta * (1 - theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_p1
                                      - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta**2 * (input_derivatives_p1
                                                 * theta**2
                                                 + 2 * input_delta
                                                 * theta_one_minus_theta
                                                 + input_derivatives
                                                 * (1 - theta)**2)
        logabsdet = tf.math.log(derivative_numerator) - \
            2 * tf.math.log(denominator)

        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (input_delta * theta**2
                                 + input_derivatives
                                 * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_p1
                                  - 2 * input_delta)
                                 * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta**2 * (input_derivatives_p1
                                             * theta**2
                                             + 2 * input_delta
                                             * theta_one_minus_theta
                                             + input_derivatives
                                             * (1 - theta)**2)
    logabsdet = tf.math.log(derivative_numerator) - \
        2 * tf.math.log(denominator)

    return outputs, logabsdet
