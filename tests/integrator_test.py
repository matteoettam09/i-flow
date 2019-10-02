""" Test spline implementation. """

import unittest

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from flow.integration.integrator import Integrator
from flow.integration.divergences import Divergence

tfd = tfp.distributions  # pylint: disable=invalid-name


def test_integrator_init():
    """ Test Integrator initialization. """
    func = unittest.mock.MagicMock()
    dist = unittest.mock.MagicMock()
    optimizer = unittest.mock.MagicMock()
    integral = Integrator(func, dist, optimizer)
    assert integral.loss_func == Divergence()('chi2')


def test_train_one_step():
    """ Test the train one step function. """
    tf.config.experimental_run_functions_eagerly(True)
    func = unittest.mock.MagicMock(return_value=tf.random.uniform([100]))
    dist = unittest.mock.MagicMock()
    dist.sample = unittest.mock.MagicMock(
        return_value=tf.ones([100]))
    dist.log_prob = unittest.mock.MagicMock(
        return_value=-4.60517*tf.ones([100]))
    dist.prob = unittest.mock.MagicMock(return_value=0.01*tf.ones([100]))
    dist.trainable_variables = [dist.prob()]
    optimizer = unittest.mock.MagicMock()
    optimizer.apply_gradients = unittest.mock.MagicMock()
    integral = Integrator(func, dist, optimizer)
    loss = integral.train_one_step(100)

    assert loss > 0

    dist.sample.assert_called_once_with(100)
    func.assert_called_once()
    dist.log_prob.assert_called_once()
    assert dist.prob.call_count == 2
    optimizer.apply_gradients.assert_called_once()


def test_integrate():
    """ Test the integrate function. """
    tf.config.experimental_run_functions_eagerly(True)
    func = unittest.mock.MagicMock(return_value=tf.random.uniform([1000]))
    dist = unittest.mock.MagicMock()
    dist.sample = unittest.mock.MagicMock(
        return_value=tf.ones([1000]))
    dist.prob = unittest.mock.MagicMock(return_value=0.001*tf.ones([1000]))
    optimizer = unittest.mock.MagicMock()
    integral = Integrator(func, dist, optimizer)
    mean, var = integral.integrate(1000)

    assert abs(mean - 1.0) < var

    dist.sample.assert_called_once_with(1000)
    dist.prob.assert_called_once()
    func.assert_called_once()


def test_acceptance():
    """ Test the integral acceptance. """
    tf.config.experimental_run_functions_eagerly(True)
    func = unittest.mock.MagicMock(return_value=tf.random.uniform([1000]))
    dist = unittest.mock.MagicMock()
    dist.sample = unittest.mock.MagicMock(
        return_value=tf.ones([1000]))
    dist.prob = unittest.mock.MagicMock(return_value=0.001*tf.ones([1000]))
    optimizer = unittest.mock.MagicMock()
    integral = Integrator(func, dist, optimizer)
    acceptance = integral.acceptance(1000)

    assert 0 < np.mean(acceptance)/np.max(acceptance) < 1

    dist.sample.assert_called_once_with(1000)
    dist.prob.assert_called_once()
    func.assert_called_once()


def test_acceptance_calc():
    """ Test the integral acceptance calculation. """
    tf.config.experimental_run_functions_eagerly(True)
    func = unittest.mock.MagicMock(return_value=0.0001*tf.ones([10000]))
    dist = unittest.mock.MagicMock()
    optimizer = unittest.mock.MagicMock()
    integral = Integrator(func, dist, optimizer)
    integral.acceptance = unittest.mock.MagicMock(
        return_value=tf.ones([10000]))
    acceptance_calc = integral.acceptance_calc(10000.**-0.5)

    assert acceptance_calc == 1

    integral.acceptance.assert_called_once_with(10000)
