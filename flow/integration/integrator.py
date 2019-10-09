""" Implement the flow integrator. """

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from . import divergences
from . import sinkhorn

# pylint: disable=invalid-name
tfb = tfp.bijectors
tfd = tfp.distributions
# pylint: enable=invalid-name


def ewma(data, window):
    """
    Function to caluclate the Exponentially weighted moving average.

    Args:
        data (np.ndarray, float64): An array of data for the average to be
                                    calculated with.
        window (int64): The decay window.

    Returns:
        int64: The EWMA for the last point in the data array
    """
    if len(data) <= window:
        return data[-1]

    wgts = np.exp(np.linspace(-1., 0., window))
    wgts /= wgts.sum()
    out = np.convolve(data, wgts, mode='full')[:len(data)]
    out[:window] = out[window]
    return out[-1]


class Integrator():
    """ Class implementing a normalizing flow integrator.

    Metrics:
    -------
        - KL-Divergence
        - Chi2-Divergence

    """

    def __init__(self, func, dist, optimizer, loss_func='chi2', **kwargs):
        """ Initialize the normalizing flow integrator. """
        self._func = func
        self.global_step = 0
        self.dist = dist
        self.optimizer = optimizer
        self.divergence = divergences.Divergence(**kwargs)
        self.loss_func = sinkhorn.sinkhorn_normalized
        self.loss_func = self.divergence(loss_func)

    @tf.function
    def train_one_step(self, nsamples, integral=False):
        """ Preform one step of integration and improve the sampling. """
        with tf.GradientTape() as tape:
            samples = tf.stop_gradient(self.dist.sample(nsamples))
            logq = self.dist.log_prob(samples)
            test = self.dist.prob(samples)
            true = tf.abs(self._func(samples))
            mean, var = tf.nn.moments(x=true/test, axes=[0])
            true = true/mean
            logp = tf.where(true > 1e-16, tf.math.log(true),
                            tf.math.log(true+1e-16))
            # loss = self.loss_func(samples, samples, 1e-2, true, test, 5)
            loss = self.loss_func(true, test, logp, logq)

        grads = tape.gradient(loss, self.dist.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.dist.trainable_variables))

        if integral:
            return loss, mean, tf.sqrt(var/nsamples)

        return loss

    @tf.function
    def sample(self, nsamples):
        """ Sample from the trained distribution. """
        return self.dist.sample(nsamples)

    @tf.function
    def integrate(self, nsamples):
        """ Integrate the function with trained distribution. """
        samples = self.dist.sample(nsamples)
        test = self.dist.prob(samples)
        true = self._func(samples)
        return tf.nn.moments(x=true/test, axes=[0])

    @tf.function
    def acceptance(self, nsamples, yield_samples=False):
        """ Calculate the acceptance for nsamples points. """
        samples = self.dist.sample(nsamples)
        test = self.dist.prob(samples)
        true = self._func(samples)

        if yield_samples:
            return true/test, samples

        return true/test

#    @tf.function
    def acceptance_calc(self, accuracy):
        """ Calculate the acceptance using a right tailed confidence interval
        with an accuracy of accuracy. """

        eta = 1
        eta_0 = 0
        weights = []
        tf_weights = tf.zeros(0)
        while abs(eta - eta_0)/eta > 0.01:
            nsamples = (tf.cast(tf.math.ceil(accuracy**-2/eta),
                                dtype=tf.int32)
                        - len(tf_weights))
            iterations = tf.ones(nsamples // 10000, dtype=tf.int32)*10000
            remainder = tf.cast(tf.range(nsamples // 10000) < nsamples % 10000,
                                tf.int32)
            iterations += remainder
            for iteration in iterations:
                weights.append(self.acceptance(iteration.numpy()))
            tf_weights = tf.sort(tf.concat(weights, axis=0))
            cum_weights = tf.cumsum(tf_weights)
            cum_weights /= cum_weights[-1]
            index = tf.cast(
                tf.searchsorted(cum_weights,
                                tf.convert_to_tensor([1-accuracy],
                                                     dtype=cum_weights.dtype)),
                dtype=tf.int32)
            max_val = tf_weights[index[0]]
            avg_val = tf.reduce_mean(tf_weights[:index[0]])
            eta_0 = eta
            eta = avg_val/max_val
            tf.print(eta_0, eta)

        return avg_val/max_val

    def save(self):
        """ Save the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.save_weights(
                './models/model_layer_{:02d}'.format(j))

    def load(self):
        """ Load the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.load_weights(
                './models/model_layer_{:02d}'.format(j))
        print("Model loaded successfully")
