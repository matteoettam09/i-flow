import tensorflow as tf
import numpy as np
from splines.spline import _padded, _knot_positions, _gather_squeeze, _search_sorted

def linear_spline(inputs, unnormalized_pdf,
                  inverse=False,
                  left=0., right=1., bottom=0., top=1.):

    if not inverse: 
        out_of_bounds = (inputs < left) | (inputs > right)
        tf.where(out_of_bounds, left, inputs)
    else:
        out_of_bounds = (inputs < bottom) | (inputs > top)
        tf.where(out_of_bounds, bottom, inputs)


    if inverse:
        inputs = (inputs - bottom) / (top - bottom)
    else:
        inputs = (inputs - left) / (right - left)

    num_bins = unnormalized_pdf.shape[-1]
    pdf = tf.nn.softmax(unnormalized_pdf, axis=-1)
    cdf = _knot_positions(pdf, 0)

    if inverse:
        inv_bin_idx = _search_sorted(cdf, inputs)
        bin_boundaries = tf.linspace(0., 1., num_bins+1)
        slopes = ((cdf[..., 1:] - cdf[..., :-1]) 
                  / (bin_boundaries[..., 1:] - bin_boundaries[..., :-1]))
        offsets = cdf[..., 1:] - slopes * bin_boundaries[..., 1:]

        input_slopes = _gather_squeeze(slopes, inv_bin_idx)
        input_offsets = _gather_squeeze(offsets, inv_bin_idx)

        outputs = (inputs - input_offsets) / input_slopes

        logabsdet = -tf.math.log(input_slopes)
    else:
        bin_pos = inputs * num_bins
        bin_idx_float = tf.floor(bin_pos)
        bin_idx = tf.cast(bin_idx_float,dtype=tf.int32)[...,tf.newaxis]

        alpha = bin_pos - bin_idx_float
        input_pdfs = _gather_squeeze(pdf, bin_idx)

        outputs = _gather_squeeze(cdf[...,:-1], bin_idx)
        outputs += alpha * input_pdfs

        bin_width = 1.0 / num_bins
        logabsdet = tf.math.log(input_pdfs) - tf.math.log(bin_width)
        
    outputs = tf.clip_by_value(outputs, 0, 1)

    if inverse:
        outputs = outputs * (right - left) + left
        logabsdet = logabsdet - tf.math.log(top - bottom) + tf.math.log(right - left)
    else:
        outputs = outputs * (top - bottom) + bottom
        logabsdet = logabsdet + tf.math.log(top - bottom) - tf.math.log(right - left)

    return outputs, logabsdet


if __name__ == '__main__':
    nbatch = 10000
    ndims = 10
    num_bins = 32

    unnormalized_pdf = np.random.random((nbatch,ndims,num_bins))

    def call_spline_fn(inputs, inverse=False):
        return linear_spline(
                inputs=inputs,
                unnormalized_pdf=unnormalized_pdf,
                inverse=inverse
        )

    inputs = np.random.random((nbatch,ndims))
    outputs, logabsdet = call_spline_fn(inputs, inverse=False)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, inverse=True)

    print(np.allclose(inputs,inputs_inv))



