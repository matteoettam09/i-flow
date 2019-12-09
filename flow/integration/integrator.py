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
        self.samples = tf.constant(self.dist.sample(1))
        self.ckpt_manager = None

    def manager(self, ckpt_manager):
        """ Set the check point manager """ 
        self.ckpt_manager = ckpt_manager

    @tf.function
    def train_one_step(self, nsamples, integral=False):
        """ Preform one step of integration and improve the sampling.

        Args:
            - nsamples: Number of samples to be taken in a training step
            - integral: Flag for returning the integral value of not.

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
    # def acceptance_calc(self, accuracy, max_samples=50000, min_samples=5000):
        # """ Calculate the acceptance using a right tailed confidence interval
        # with an accuracy of accuracy. """

        # # @tf.function
        # def _calc_efficiency(weights):
            # weights = tf.convert_to_tensor(weights, dtype=tf.float64)
            # weights = tf.sort(weights)
            # cum_weights = tf.cumsum(weights)
            # cum_weights /= cum_weights[-1]
            # index = tf.cast(
                # tf.searchsorted(cum_weights,
                                # tf.convert_to_tensor([1-accuracy],
                                                     # dtype=tf.float64)),
                # dtype=tf.int32)
            # max_val = weights[index[0]]
            # avg_val = tf.reduce_mean(weights[:index[0]])
            # return avg_val, max_val

        # eta = 1
        # eta_0 = 0
        # weights = []
        # while abs(eta - eta_0)/eta > 0.01:
            # eta_0 = eta
            # nsamples = np.ceil(accuracy**-2/eta).astype(np.int64)-len(weights)
            # nsamples = np.min([nsamples, max_samples])
            # nsamples = np.max([nsamples, min_samples])
            # weights.extend(self.acceptance(nsamples))
            # avg_val, max_val = _calc_efficiency(weights)
            # eta = avg_val/max_val
            # print(eta_0, eta)

        # return avg_val, max_val

    def acceptance_calc(self, accuracy, max_samples=50000, min_samples=5000):
        """ Calculate the acceptance using a right tailed confidence interval
        with an accuracy of accuracy.

        Args:
            accuracy (float): Desired accuracy for total cross-section
            max_samples (int): Max number of samples per iteration
            min_samples (int): Min number of samples per iteration

        Returns:
            (tuple): tuple containing:

                avg_val (float): Average weight value from all iterations
                max_val (float): Maximum value to use in unweighting

        """

        # @tf.function
        def _calc_efficiency(weights):
            weights = tf.convert_to_tensor(weights, dtype=tf.float64)
            weights = tf.sort(weights)
            i_max = tf.convert_to_tensor([ int(np.ceil(len(weights)*(1-accuracy)))], dtype=tf.int32)
            max_val = weights[i_max[0]]
            avg_val = tf.reduce_mean(weights[:i_max[0]])
            return avg_val, max_val

        weights = []
        precision=0.1
        NSAMP = (1./precision)**2 / accuracy

        while len(weights) < NSAMP:
            nsamples = min_samples
            _w = self.acceptance(nsamples)
            weights.extend([w for w in _w if w !=0])
            avg_val, max_val = _calc_efficiency(weights)
            eta = avg_val/max_val
            print(eta, nsamples, len(weights), NSAMP)

        del weights

        return avg_val, max_val
        # weights = []
        # while abs(eta - eta_0)/eta > 0.01:
            # eta_0 = eta
            # nsamples = np.ceil(accuracy**-2/eta).astype(np.int64)-len(weights)
            # nsamples = np.min([nsamples, max_samples])
            # nsamples = np.max([nsamples, min_samples])
            # weights.extend(self.acceptance(nsamples))
            # avg_val, max_val = _calc_efficiency(weights)
            # eta = avg_val/max_val
            # print(eta_0, eta, nsamples, len(weights))

        # return avg_val, max_val

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
        """ Function to load a checkpoint of the model and optimizer,
            as well as any other trackables in the checkpoint.
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
