""" Implement the flow integrator. """

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from . import divergences
# from . import sinkhorn

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
        float64: The EWMA for the last point in the data array
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

    Args:
        - func: Function to be integrated
        - dist: Distribution to be trained to match the function
        - optimizer: An optimizer from tensorflow used to train the network
        - loss_func: The loss function to be minimized
        - kwargs: Additional arguments that need to be passed to the loss

    """
    def __init__(self, func, dist, optimizer, loss_func='chi2', **kwargs):
        """ Initialize the normalizing flow integrator. """
        self._func = func
        self.global_step = 0
        self.dist = dist
        self.optimizer = optimizer
        self.divergence = divergences.Divergence(**kwargs)
        # self.loss_func = sinkhorn.sinkhorn_loss
        self.loss_func = self.divergence(loss_func)
        # self.samples = tf.constant(self.dist.sample(1))
        self.ckpt_manager = None

    def manager(self, ckpt_manager):
        """ Set the check point manager """
        self.ckpt_manager = ckpt_manager

    @tf.function
    def train_one_step(self, nsamples, integral=False):
        """ Perform one step of integration and improve the sampling.

        Args:
            - nsamples(int): Number of samples to be taken in a training step
            - integral(bool): Flag for returning the integral value or not.

        Returns:
            - loss: Value of the loss function for this step
            - integral (optional): Estimate of the integral value
            - uncertainty (optional): Integral statistical uncertainty

        """
        samples = self.dist.sample(nsamples)
        # self.samples = tf.concat([self.samples, samples], 0)
        # if self.samples.shape[0] > 5001:
        #     self.samples = self.samples[nsamples:]
        true = tf.abs(self._func(samples))
        with tf.GradientTape() as tape:
            test = self.dist.prob(samples)
            logq = self.dist.log_prob(samples)
            mean, var = tf.nn.moments(x=true/test, axes=[0])
            true = tf.stop_gradient(true/mean)
            logp = tf.where(true > 1e-16, tf.math.log(true),
                            tf.math.log(true+1e-16))
            # loss = self.loss_func(samples, samples, 1e-1, true, test, 100)
            loss = self.loss_func(true, test, logp, logq)

        grads = tape.gradient(loss, self.dist.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.dist.trainable_variables))

        if integral:
            return loss, mean, tf.sqrt(var/nsamples)

        return loss

    @tf.function
    def sample(self, nsamples):
        """ Sample from the trained distribution.

        Args:
            nsamples(int): Number of points to be sampled.

        Returns:
            tf.tensor of size (nsamples, ndim) of sampled points.

        """
        return self.dist.sample(nsamples)

    @tf.function
    def integrate(self, nsamples):
        """ Integrate the function with trained distribution.

        This method estimates the value of the integral based on
        Monte Carlo importance sampling. It returns a tuple of two
        tf.tensors. The first one is the mean, i.e. the estimate of
        the integral. The second one gives the variance of the integrand.
        To get the variance of the estimated mean, the returned variance
        needs to be divided by (nsamples -1).

        Args:
            nsamples(int): Number of points on which the estimate is based on.

        Returns:
            tuple of 2 tf.tensors: mean and variance

        """
        samples = self.dist.sample(nsamples)
        test = self.dist.prob(samples)
        true = self._func(samples)
        return tf.nn.moments(x=true/test, axes=[0])

    @tf.function
    def sample_weights(self, nsamples, yield_samples=False):
        """ Sample from the trained distribution and return their weights.

        This method samples 'nsamples' points from the trained distribution
        and computes their weights, defined as the functional value of the
        point divided by the probability of the trained distribution of
        that point.

        Optionally, the drawn samples can be returned, too.

        Args:
            nsamples (int): Number of samples to be drawn.
            yield_samples (bool): Also return samples if true.

        Returns:
            true/test: tf.tensor of size (nsamples, 1) of sampled weights
            (samples: tf.tensor of size (nsamples, ndims) of sampled points)

        """
        samples = self.dist.sample(nsamples)
        test = self.dist.prob(samples)
        true = self._func(samples)

        if yield_samples:
            return true/test, samples

        return true/test

    def acceptance(self, nopt, npool=50, nreplica=1000):
        """ Calculate the acceptance, i.e. the unweighting
            efficiency as discussed in
            "Event Generation with Normalizing Flows"
            by C. Gao, S. Hoeche, J. Isaacson, C. Krause and H. Schulz

        Args:
            nopt (int): Number of points on which the optimization was based on.
            npool (int): called n in the reference
            nreplica (int): called m in the reference

        Returns:
            (float): unweighting efficiency

        """

        weights = []
        for _ in range(npool):
            wgt = self.sample_weights(nopt)
            weights.append(wgt)
        weights = np.concatenate(weights)

        sample = np.random.choice(weights, (nreplica, nopt))
        s_max = np.max(sample, axis=1)
        s_mean = np.mean(sample, axis=1)
        s_acc = np.mean(s_mean) / np.median(s_max)

        return s_acc

#    def acceptance_calc(self, accuracy, max_samples=50000, min_samples=5000):
#        """ Calculate the acceptance using a right tailed confidence interval
#        with an accuracy of accuracy.
#
#        This method is deprecated and will be removed in the future.
#
#
#        Args:
#            accuracy (float): Desired accuracy for total cross-section
#            max_samples (int): Max number of samples per iteration
#            min_samples (int): Min number of samples per iteration
#
#        Returns:
#            (tuple): tuple containing:
#
#                avg_val (float): Average weight value from all iterations
#                max_val (float): Maximum value to use in unweighting
#
#        """
#
#        # @tf.function
#        def _calc_efficiency(weights):
#            weights = tf.convert_to_tensor(weights, dtype=tf.float64)
#            weights = tf.sort(weights)
#            i_max = tf.convert_to_tensor([
#                int(np.ceil(len(weights)*(1-accuracy)))], dtype=tf.int32)
#            max_val = weights[i_max[0]]
#            avg_val = tf.reduce_mean(weights[:i_max[0]])
#            return avg_val, max_val
#
#        weights = []
#        precision=0.1
#        NSAMP = (1./precision)**2 / accuracy
#
#        while len(weights) < NSAMP:
#            nsamples = min_samples
#            _w = self.acceptance(nsamples)
#            weights.extend([w for w in _w if w !=0])
#            avg_val, max_val = _calc_efficiency(weights)
#            eta = avg_val/max_val
#            print(eta, nsamples, len(weights), NSAMP)
#
#        del weights
#
#        return avg_val, max_val

    def save_weights(self):
        """ Save the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.save_weights(
                './models/model_layer_{:02d}'.format(j))

    def load_weights(self):
        """ Load the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.load_weights(
                './models/model_layer_{:02d}'.format(j))
        print("Model loaded successfully")

    def save(self):
        """ Function to save a checkpoint of the model and optimizer,
            as well as any other trackables in the checkpoint.
            Note that the network architecture is not saved, so the same
            network architecture must be used for the same saved and loaded
            checkpoints (network arch can be saved if required).
        """

        if self.ckpt_manager is not None:
            save_path = self.ckpt_manager.save()
            print("Saved checkpoint at: {}".format(save_path))
        else:
            print("There is no checkpoint manager supplied for saving the "
                  "network weights, optimizer, or other trackables.")
            print("Therefore these will not be saved and the training will "
                  "start from default values in the future.")
            print("Consider using a checkpoint manager to save the network "
                  "weights and optimizer.")

    def load(self, loadname, checkpoint=None):
        """ Function to load a checkpoint of the model, optimizer,
            and any other trackables in the checkpoint.

            Note that the network architecture is not saved, so the same
            network architecture must be used for the same saved and loaded
            checkpoints. Network arch can be loaded if it is saved.

        Args:
            loadname (str) : The postfix of the directory where the checkpoints
                             are saved, e.g.,
                             ckpt_dir = "./models/tf_ckpt_" + loadname + "/"
            checkpoint (object): tf.train.checkpoint instance.
        Returns:
            Nothing returned.

        """
        ckpt_dir = "./models/tf_ckpt_" + loadname + "/"
        if checkpoint is not None:
            status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
            status.assert_consumed()
            print("Loaded checkpoint")
        else:
            print("Not Loading any checkpoint")
            print("Starting training from initial configuration")
