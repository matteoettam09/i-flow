import tensorflow as tf

def _padded(t, lhs, rhs=None):
    """Left pads and optionally right pads the innermost axis of `t`."""
    lhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
    zeros = tf.zeros([tf.rank(t) - 1, 2], dtype=tf.int32)
    lhs_paddings = tf.concat([zeros, [[1, 0]]], axis=0)
    result = tf.pad(t, paddings=lhs_paddings, constant_values=lhs)
    if rhs is not None:
        rhs = tf.convert_to_tensor(lhs, dtype=t.dtype)
        rhs_paddings = tf.concat([zeros, [[0, 1]]], axis=0)
        result = tf.pad(result, paddings=rhs_paddings, constant_values=rhs)
    return result

def _knot_positions(bin_sizes, range_min):
    return _padded(tf.cumsum(bin_sizes, axis=-1) + range_min, lhs=range_min)

def _gather_squeeze(params, indices):
    rank = len(indices.shape)
    if rank is None:
        raise ValueError('`indices` must have a statically known rank.')
    return tf.gather(params, indices, axis=-1, batch_dims=rank - 1)[..., 0]

def _search_sorted(cdf, inputs):
    return tf.maximum(tf.zeros([], dtype=tf.int32),
                tf.searchsorted(
                    cdf[..., :-1], 
                    inputs[...,tf.newaxis],
                    side='right',
                    out_type=tf.int32) - 1)

def _cube_root(x):
    return tf.sign(x) * tf.exp(tf.math.log(tf.abs(x))/3.0)
