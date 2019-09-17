""" Implement tests for the integrator class. """

# pylint: disable=invalid-name, protected-access

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from flow.integration.integrator import Integrator

tfd = tfp.distributions

tf.keras.backend.set_floatx('float64')


def _func(x):
    return x


def test_integrator_init():
    """ Test integrator initialization. """
    dist = tfd.Uniform(low=3*[0.], high=3*[1.])
    optimizer = tf.keras.optimizers.Adam(1e-3)
    integrator = Integrator(_func, dist, optimizer)

    assert integrator.global_step == 0
    assert integrator._func(1) == _func(1)
    assert isinstance(integrator.dist, tfd.Uniform)
    assert isinstance(integrator.optimizer, tf.keras.optimizers.Adam)


def test_one_step():
    """ Test training one step. """
    dist = tfd.Uniform(low=3*[0.], high=3*[1.])
    optimizer = tf.keras.optimizers.Adam(1e-3)
    integrator = Integrator(_func, dist, optimizer)

    loss, integral, std = integrator.train_one_step(10, integral=True)

    assert np.all(loss > 0)
    assert np.all(integral > 0)
    assert np.all(std > 0)
