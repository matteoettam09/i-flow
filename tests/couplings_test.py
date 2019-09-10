import pytest

from flow.integration import couplings
import numpy as np
import tensorflow as tf

import itertools

tf.keras.backend.set_floatx('float64')

def build_dense(in_features, out_features):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(in_features))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(out_features))
    model.summary()
    return model

def test_coupling_init():
    with pytest.raises(NotImplementedError):
        couplings.CouplingBijector([1,1,0,0], build_dense)

    with pytest.raises(ValueError):
        couplings.CouplingBijector([1,1,0,0], build_dense, blob='a')

def test_affine_init():
    layer = couplings.AffineBijector([1,1,0,0], build_dense)
    
    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 2
    assert layer.transform_net.layers[-1].output_shape[-1] == 4

    layer = couplings.AffineBijector([1,1,0,0], build_dense, blob=10)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 20
    assert layer.transform_net.layers[-1].output_shape[-1] == 4

def test_affine_inversion():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.AffineBijector([1,1,0,0], build_dense)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

    layer = couplings.AffineBijector([1,1,0,0], build_dense, blob=10)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

def test_affine_determinant():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.AffineBijector([1,1,0,0], build_dense)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.AffineBijector([1,1,0,0], build_dense, blob=10)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

def test_additive_init():
    layer = couplings.AdditiveBijector([1,1,0,0], build_dense)
    
    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 2
    assert layer.transform_net.layers[-1].output_shape[-1] == 2

    layer = couplings.AdditiveBijector([1,1,0,0], build_dense, blob=10)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 20
    assert layer.transform_net.layers[-1].output_shape[-1] == 2

def test_additive_inversion():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.AdditiveBijector([1,1,0,0], build_dense)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

    layer = couplings.AdditiveBijector([1,1,0,0], build_dense, blob=10)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

def test_additive_determinant():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.AdditiveBijector([1,1,0,0], build_dense)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.AdditiveBijector([1,1,0,0], build_dense, blob=10)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

def test_linear_init():
    layer = couplings.PiecewiseLinear([1,1,0,0], build_dense)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 2
    assert layer.transform_net.layers[-1].output_shape[-1] == 20

    layer = couplings.PiecewiseLinear([1,1,0,0], build_dense, blob=10)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 20
    assert layer.transform_net.layers[-1].output_shape[-1] == 20

def test_linear_inversion():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseLinear([1,1,0,0], build_dense)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

    layer = couplings.PiecewiseLinear([1,1,0,0], build_dense, blob=10)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

def test_linear_determinant():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseLinear([1,1,0,0], build_dense)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseLinear([1,1,0,0], build_dense, blob=10)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

def test_linear_masks():
    mask = [1,0,0]
    masks = itertools.permutations(mask)
    inputs = np.array(np.random.random((100,len(mask))),dtype=np.float64)
    for _mask in masks:
        layer = couplings.PiecewiseLinear(_mask, build_dense)
        assert (inputs == layer.inverse(layer.forward(inputs)).numpy()).all()

def test_quadratic_init():
    layer = couplings.PiecewiseQuadratic([1,1,0,0], build_dense)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 2
    assert layer.transform_net.layers[-1].output_shape[-1] == 2*(2*10+1)

    layer = couplings.PiecewiseQuadratic([1,1,0,0], build_dense, blob=10)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 20
    assert layer.transform_net.layers[-1].output_shape[-1] == 2*(2*10+1)

def test_quadratic_inversion():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseQuadratic([1,1,0,0], build_dense)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

    layer = couplings.PiecewiseQuadratic([1,1,0,0], build_dense, blob=10)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

def test_quadratic_determinant():
    inputs = np.array(np.random.random((10,4)),dtype=np.float64)
    layer = couplings.PiecewiseQuadratic([1,1,0,0], build_dense)

    print(layer._forward_log_det_jacobian(inputs))
    print(layer._inverse_log_det_jacobian(layer._forward(inputs)))
    print(layer._forward_log_det_jacobian(inputs) +
          layer._inverse_log_det_jacobian(layer._forward(inputs)))
    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

def test_quadratic_masks():
    mask = [1,0,0]
    masks = itertools.permutations(mask)
    inputs = np.array(np.random.random((100,len(mask))),dtype=np.float64)
    for _mask in masks:
        layer = couplings.PiecewiseQuadratic(_mask, build_dense)
        assert (inputs == layer.inverse(layer.forward(inputs)).numpy()).all()

def test_cubic_init():
    layer = couplings.PiecewiseCubic([1,1,0,0], build_dense)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 2
    assert layer.transform_net.layers[-1].output_shape[-1] == 2*(2*10+2)

    layer = couplings.PiecewiseCubic([1,1,0,0], build_dense, blob=10)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 20
    assert layer.transform_net.layers[-1].output_shape[-1] == 2*(2*10+2)

def test_cubic_inversion():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseCubic([1,1,0,0], build_dense)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

    layer = couplings.PiecewiseCubic([1,1,0,0], build_dense, blob=10)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

def test_cubic_determinant():
    inputs = np.array(np.random.random((10,4)),dtype=np.float64)
    layer = couplings.PiecewiseCubic([1,1,0,0], build_dense)

    print(layer._forward_log_det_jacobian(inputs))
    print(layer._inverse_log_det_jacobian(layer._forward(inputs)))
    print(layer._forward_log_det_jacobian(inputs) +
          layer._inverse_log_det_jacobian(layer._forward(inputs)))
    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

def test_cubic_masks():
    mask = [1,0,0]
    masks = itertools.permutations(mask)
    inputs = np.array(np.random.random((100,len(mask))),dtype=np.float64)
    for _mask in masks:
        layer = couplings.PiecewiseCubic(_mask, build_dense)
        assert (inputs == layer.inverse(layer.forward(inputs)).numpy()).all()

def test_rational_quadratic_init():
    layer = couplings.PiecewiseRationalQuadratic([1,1,0,0], build_dense)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 2
    assert layer.transform_net.layers[-1].output_shape[-1] == 2*(10*3+1)

    layer = couplings.PiecewiseRationalQuadratic([1,1,0,0], build_dense, blob=10)

    assert layer.num_identity_features == 2
    assert layer.num_transform_features == 2
    assert layer.transform_net.input_spec.axes[-1] == 20
    assert layer.transform_net.layers[-1].output_shape[-1] == 2*(10*3+1)

def test_rational_quadratic_inversion():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseRationalQuadratic([1,1,0,0], build_dense)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

    layer = couplings.PiecewiseRationalQuadratic([1,1,0,0], build_dense, blob=10)
    assert np.allclose(inputs, layer.inverse(layer.forward(inputs)))
    assert np.allclose(inputs, layer.forward(layer.inverse(inputs)))

def test_rational_quadratic_determinant():
    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseRationalQuadratic([1,1,0,0], build_dense)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

    inputs = np.array(np.random.random((100,4)),dtype=np.float64)
    layer = couplings.PiecewiseRationalQuadratic([1,1,0,0], build_dense, blob=10)

    assert np.allclose(layer._forward_log_det_jacobian(inputs),
                      -layer._inverse_log_det_jacobian(layer.forward(inputs)))

def test_rational_quadratic_masks():
    mask = [1,0,0]
    masks = itertools.permutations(mask)
    inputs = np.array(np.random.random((100,len(mask))),dtype=np.float64)
    for _mask in masks:
        layer = couplings.PiecewiseRationalQuadratic(_mask, build_dense)
        assert (inputs == layer.inverse(layer.forward(inputs)).numpy()).all()
