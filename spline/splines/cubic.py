import tensorflow as tf
import numpy as np
from flow.splines.spline import _padded, _knot_positions, _gather_squeeze, _search_sorted

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_EPS = 1e-5
DEFAULT_QUADRATIC_THRESHOLD = 1e-3

def cubic_spline(inputs,
                 unnormalized_widths,
                 unnormalized_heights,
                 unnorm_derivatives_left,
                 unnorm_derivatives_right,
                 inverse=False,
                 left=0., right=1., bottom=0., top=1.,
                 min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                 min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                 eps=DEFAULT_EPS,
                 quadratic_threshold=DEFAULT_QUADRATIC_THRESHOLD):

    if not inverse and (tf.math.reduce_min(inputs) < left or tf.math.reduce_max(inputs) > right):
        raise ValueError('Outside domain')
    elif inverse and (tf.math.reduce_min(inputs) < bottom or tf.math.reduce_max(inputs) > top):
        raise ValueError('Outside domain')

    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = tf.nn.softmax(unnormalized_widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = _knot_positions(widths,0)

    heights = tf.nn.softmax(unnormalized_heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = _knot_positions(heights,0)

    slopes = heights / widths
    min_slope_1 = np.minimum(np.abs(slopes[..., :-1]),
                             np.abs(slopes[..., 1:]))
    min_slope_2 = (
            0.5 * (widths[..., 1:] * slopes[..., :-1] + widths[..., :-1] * slopes[..., 1:])
            / (widths[..., :-1] + widths[..., 1:])
    )
    min_slope = np.minimum(min_slope_1, min_slope_2)

    derivatives_left = tf.nn.sigmoid(unnorm_derivatives_left) * 3 * slopes[..., 0][..., tf.newaxis]
    derivatives_right = tf.nn.sigmoid(unnorm_derivatives_right) * 3 * slopes[..., -1][..., tf.newaxis]

    derivatives = min_slope * (tf.sign(slopes[..., :-1]) + tf.sign(slopes[..., 1:]))
    derivatives = tf.concat([derivatives_left,
                             derivatives,
                             derivatives_right], axis=-1)

    a = (derivatives[..., :-1] + derivatives[..., 1:] - 2 * slopes) / widths**2
    b = (3 * slopes - 2 * derivatives[..., :-1] - derivatives[..., 1:]) / widths
    c = derivatives[..., :-1]
    d = cumheights[..., :-1]

    if inverse:
        bin_idx = _search_sorted(cumheights, inputs)[...,tf.newaxis]
    else:
        bin_idx = _search_sorted(cumwidths, inputs)[...,tf.newaxis]

    inputs_a = _gather_squeeze(a, bin_idx)
    inputs_b = _gather_squeeze(b, bin_idx)
    inputs_c = _gather_squeeze(c, bin_idx)
    inputs_d = _gather_squeeze(d, bin_idx)

    input_left_cumwidths = _gather_squeeze(cumwidths, bin_idx)
    input_right_cumwidths = _gather_squeeze(cumwidths, bin_idx+1)

    if inverse:
        # Modified coefficients for solving the cubic.
        inputs_b_ = (inputs_b / inputs_a) / 3.
        inputs_c_ = (inputs_c / inputs_a) / 3.
        inputs_d_ = (inputs_d - inputs) / inputs_a

        delta_1 = -inputs_b_**2 + inputs_c_
        delta_2 = -inputs_c_ * inputs_b_ + inputs_d_
        delta_3 = inputs_b_ * inputs_d_ - inputs_c_**2

        discriminant = 4. * delta_1 * delta_3 - delta_2**2

        depressed_1 = -2 * inputs_b_ * delta_1 + delta_2
        depressed_2 = delta_1

        three_roots_mask = discriminant >= 0 # Discriminant == 0 might be a problem in practice.
        one_root_mask = discriminant < 0

        outputs = tf.zeros_like(inputs)

        # Deal with one root cases
        p = ((-depressed_1[one_root_mask] + tf.sqrt(-discriminant[one_root_mask])) / 2.)**(1./3.)
        q = ((-depressed_1[one_root_mask] - tf.sqrt(-discriminant[one_root_mask])) / 2.)**(1./3.)

        outputs[one_root_mask] = ((p+q)
                                 - inputs_b_[one_root_mask]
                                 + input_left_cumwidths[one_root_mask])

        # Deal with three root cases
        theta = tf.atan2(tf.sqrt(discriminant[three_roots_mask]), -depressed_1[three_roots_mask])
        theta /= 3.

        cubic_root_1 = tf.cos(theta)
        cubic_root_2 = tf.sin(theta)

        root_1 = cubic_root_1
        root_2 = -0.5 * cubic_root_1 - 0.5 * tf.sqrt(3) * cubic_root_2
        root_2 = -0.5 * cubic_root_1 + 0.5 * tf.sqrt(3) * cubic_root_2

        root_scale = 2 * tf.sqrt(-depressed_2[three_roots_mask])
        root_shift = (-inputs_b_[three_roots_mask] + input_left_cumwidths[three_roots_mask])

        root_1 = root_1 * root_scale + root_shift
        root_2 = root_2 * root_scale + root_shift
        root_3 = root_3 * root_scale + root_shift

        root1_mask = tf.cast((input_left_cumwidths[three_roots_mask] - eps) < root_1, dtype=tf.float32)
        root1_mask *= tf.cast(root1 < (input_right_cumwidths[three_roots_mask] + eps), dtype=tf.float32)

        root2_mask = tf.cast((input_left_cumwidths[three_roots_mask] - eps) < root_1, dtype=tf.float32)
        root2_mask *= tf.cast(root2 < (input_right_cumwidths[three_roots_mask] + eps), dtype=tf.float32)

        root3_mask = tf.cast((input_left_cumwidths[three_roots_mask] - eps) < root_1, dtype=tf.float32)
        root3_mask *= tf.cast(root3 < (input_right_cumwidths[three_roots_mask] + eps), dtype=tf.float32)

        roots = tf.stack([root_1, root_2, root_3], axis=-1)
        masks = tf.stack([root1_mask, root2_mask, root3_mask], axis=-1)
        mask_index = tf.argsort(masks, axis=-1, descending=True)[..., 0][..., tf.newaxis]
        outputs[three_roots_mask] = _gather_squeeze(roots, mask_index)

        # Deal with a -> 0 (almost quadratic) cases
        
        quadratic_mask = tf.abs(inputs_a) < quadratic_threshold
        a = inputs_b[quadratic_mask]
        b = inputs_c[quadratic_mask]
        c = (inputs_d[quadratic_mask] - inputs[quadratic_mask])
        alpha = (-b + tf.sqrt(b**2 - 4*a*c)) / (2*a)
        outputs[quadratic_mask] = alpha + input_left_cumwidths[quadratic_mask]

        shifted_outputs = (outputs - input_left_cumwidths)
        logabsdet = -np.log(3 * inputs_a * shifted_outputs ** 2 
                          + 2 * inputs_b * shifted_outputs
                          + inputs_c)

    else:
        shifted_inputs = (inputs - input_left_cumwidths)
        outputs = (inputs_a * shifted_inputs**3
                 + inputs_b * shifted_inputs**2
                 + inputs_c * shifted_inputs
                 + inputs_d)

        logabsdet = np.log(3 * inputs_a * shifted_inputs ** 2 
                         + 2 * inputs_b * shifted_inputs
                         + inputs_c)

    outputs = tf.clip_by_value(outputs, 0, 1)

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - np.log(top - bottom) + np.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + np.log(top - bottom) - np.log(right - left)

    return outputs, logabsdet
        
if __name__ == '__main__':
    nbatch = 10000
    ndims = 10
    num_bins = 32

    unnormalized_widths = np.random.random((nbatch,ndims,num_bins))
    unnormalized_heights = np.random.random((nbatch,ndims,num_bins))
    unnorm_derivatives_left = np.random.random((nbatch,ndims,num_bins))
    unnorm_derivatives_right = np.random.random((nbatch,ndims,num_bins))

    def call_spline_fn(inputs, inverse=False):
        return cubic_spline(
                inputs=inputs,
                unnormalized_widths=unnormalized_widths,
                unnormalized_heights=unnormalized_heights,
                unnorm_derivatives_left=unnorm_derivatives_left,
                unnorm_derivatives_right=unnorm_derivatives_right,
                inverse=inverse
        )

    inputs = np.random.random((nbatch,ndims))
    outputs, logabsdet = call_spline_fn(inputs, inverse=False)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

    print(np.allclose(inputs,inputs_inv))
