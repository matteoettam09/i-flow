""" Learn an easy function, e.g. a 2d Gaussian
    and 'blind-out' a slice to see if i-flow still
    learns it correctly.
    This is a first test for i-flow as density estimator.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import erf

from absl import app, flags

from iflow.integration import integrator
from iflow.integration import couplings 

tfd = tfp.distributions  # pylint: disable=invalid-name
tfb = tfp.bijectors  # pylint: disable=invalid-name
tf.keras.backend.set_floatx('float64')

FLAGS = flags.FLAGS
flags.DEFINE_float('alpha', 0.2, 'The width of the Gaussian',
                   short_name='a')
flags.DEFINE_float('cut1', 0.6, 'Left edge of cut region in x1',
                   short_name='c1')
flags.DEFINE_float('cut2', 0.75, 'Right edge of cut region in x2',
                   short_name='c2')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('ptspepoch', 5000, 'Number of points to sample per epoch',
                     short_name='p')
flags.DEFINE_bool('flat', False, 'Flag to sample from learned or flat distribution during training', short_name='f')

class GaussFunction:
    """ Contains the function to be used in testing """
    def __init__(self, alpha, ndims=2):
        self.ndims = ndims
        self.alpha = alpha

    def __call__(self, x):
        pre = tf.cast(1.0/(self.alpha * tf.sqrt(np.pi))**self.ndims,
                      dtype=tf.float64)
        exponent = -1.0*tf.reduce_sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        norm = erf(1/(2*self.alpha)) ** self.ndims
        return pre * tf.exp(exponent)/norm

class DensityEstimator(integrator.Integrator):
    """ i-flow Integrator class with new functions """

    def train_these_points(self, pts, nsamples, integral=False):
        """ Perform one step of integration and improve the sampling.

        Args:
            - pts(array): List of points to be used
            - integral(bool): Flag for returning the integral value or not.

        Returns:
            - loss: Value of the loss function for this step
            - integral (optional): Estimate of the integral value
            - uncertainty (optional): Integral statistical uncertainty

        """
        #samples = self.samples
        samples = pts
        true = tf.abs(self._func(samples))
        with tf.GradientTape() as tape:
            test = self.dist.prob(samples)
            logq = self.dist.log_prob(samples)
            mean, var = tf.nn.moments(x=true/test, axes=[0])
            true = tf.stop_gradient(true/mean)
            logp = tf.where(true > 1e-16, tf.math.log(true),
                            tf.math.log(true+1e-16))
            loss = self.loss_func(true, test, logp, logq)

        grads = tape.gradient(loss, self.dist.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.dist.trainable_variables))

        if integral:
            return loss, mean, tf.sqrt(var/(nsamples-1.))

        return loss


def build(in_features, out_features, options):
    """ Builds a dense NN.

    The output layer is initialized to 0, so the first pass
    before training gives the identity transformation.

    Arguments:
        in_features (int): dimensionality of the inputs space
        out_features (int): dimensionality of the output space
        options: additional arguments, not used at the moment

    Returns:
        A tf.keras.models.Model instance

    """
    del options

    invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
    hidden = tf.keras.layers.Dense(32, activation='relu')(invals)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_features, bias_initializer='zeros',
                                    kernel_initializer='zeros')(hidden)
    model = tf.keras.models.Model(invals, outputs)
    model.summary()
    return model


def mask_flip(mask):
    """ Interchange 0 <-> 1 in the mask. """
    return 1-mask


def binary_list(inval, length):
    """ Convert x into a binary list of length l. """
    return np.array([int(i) for i in np.binary_repr(inval, length)])


def binary_masks(ndims):
    """ Create binary masks for to account for symmetries. """
    n_masks = int(np.ceil(np.log2(ndims)))
    sub_masks = np.transpose(np.array(
        [binary_list(i, n_masks)
         for i in range(ndims)]))[::-1]
    flip_masks = mask_flip(sub_masks)

    # Combine masks
    masks = np.empty((2*n_masks, ndims))
    masks[0::2] = flip_masks
    masks[1::2] = sub_masks

    return masks


def build_iflow(func, ndims):
    """ Build the iflow integrator

    Args:
        func: integrand
        ndims (int): dimensionality of the integrand

    Returns: Integrator: iflow Integrator object

    """
    masks = binary_masks(ndims)
    bijector = []
    for mask in masks:
        bijector.append(couplings.PiecewiseRationalQuadratic(mask, build,
                                                             num_bins=16,
                                                             blob=None,
                                                             options=None))
    bijector = tfb.Chain(list(reversed(bijector)))
    low = np.zeros(ndims, dtype=np.float64)
    high = np.ones(ndims, dtype=np.float64)
    dist = tfd.Uniform(low=low, high=high)
    dist = tfd.Independent(distribution=dist,
                           reinterpreted_batch_ndims=1)
    dist = tfd.TransformedDistribution(
        distribution=dist,
        bijector=bijector)

    optimizer = tf.keras.optimizers.Adam(1e-3, clipnorm=10.0)
    integrate = DensityEstimator(func, dist, optimizer,
                                 loss_func='exponential')

    return integrate


def train_iflow(integrate, ptspepoch, epochs):
    """ Run the iflow integrator

    Args:
        integrate (Integrator): iflow Integrator class object
        ptspepoch (int): number of points per epoch in training
        epochs (int): number of epochs for training

    Returns:
        numpy.ndarray(float): value of loss (mean) and its uncertainty (standard deviation)

    """
    means = np.zeros(epochs+1)
    stddevs = np.zeros(epochs+1)
    for epoch in range(epochs+1):
        if FLAGS.flat:
            pts_for_training = tf.random.uniform(shape=[ptspepoch,2], dtype=tf.dtypes.float64) 
        else:
            pts_for_training = integrate.sample(ptspepoch)
        out_of_window = tf.where(tf.math.logical_or(pts_for_training[...,0]<FLAGS.cut1, pts_for_training[...,0]>FLAGS.cut2))
        pts_for_training_used = tf.gather_nd(pts_for_training, out_of_window)
        npts = tf.size(pts_for_training_used, out_type=tf.dtypes.float64)
        loss, integral, error = integrate.train_these_points(pts_for_training_used,
                                                             npts, integral=True)
        
        means[epoch] = integral
        stddevs[epoch] = error
        if epoch % 10 == 0:
            print('Epoch: {:3d} Loss = {:8e} Integral = '
                  '{:8e} +/- {:8e}'.format(epoch, loss, integral, error))
            fig = plt.figure(dpi=150, figsize=[8., 8.])
            axis = fig.add_subplot(111)
            axis.set_aspect('equal')
            plt.scatter(pts_for_training[:, 0], pts_for_training[:, 1], s=1, color='r')
            plt.scatter(pts_for_training_used[:, 0], pts_for_training_used[:, 1], s=1, color='k')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            if FLAGS.flat:
                plt.savefig('plots/gauss_flat_{:03d}.png'.format(epoch))
            else:
                plt.savefig('plots/gauss_iflow_{:03d}.png'.format(epoch))
            plt.close()

    return means, stddevs


def sample_iflow(integrate, ptspepoch, epochs):
    """ Sample from the iflow integrator

    Args:
        integrate (Integrator): iflow Integrator class object
        ptspepoch (int): number of points per epoch in training
        epochs (int): number of epochs for training

    Returns:
        (tuple): mean and stddev numpy arrays

    """
    # mean and stddev of trained NF
    print('Estimating integral from trained network')
    means_t = []
    stddevs_t = []
    for _ in range(epochs+1):
        mean, var = integrate.integrate(ptspepoch)
        means_t.append(mean)
        stddevs_t.append(tf.sqrt(var/(ptspepoch-1.)).numpy())
    return np.array(means_t), np.array(stddevs_t)


def main(argv):
    """ Main function for test runs. """
    del argv

    alpha = FLAGS.alpha
    cut1 = FLAGS.cut1
    cut2 = FLAGS.cut2

    if cut1 > cut2:
        cut1, cut2 = cut2, cut1

    func = GaussFunction(alpha)

    epochs = FLAGS.epochs
    ptspepoch = FLAGS.ptspepoch

    integrate = build_iflow(func, 2)

    mean, error = train_iflow(integrate, ptspepoch, epochs)
    mean, error = sample_iflow(integrate, ptspepoch*10, 10)
    iflow_mean_wgt = np.sum(mean/(error**2), axis=-1)
    iflow_err_wgt = np.sum(1./(error**2), axis=-1)
    iflow_mean_wgt /= iflow_err_wgt
    iflow_err_wgt = 1. / np.sqrt(iflow_err_wgt)

    print("Weighted result is {:.5e} +/- {:.5e}".format(
        iflow_mean_wgt, iflow_err_wgt))

    fig = plt.figure(dpi=150, figsize=[8., 8.])
    axis = fig.add_subplot(111)
    axis.set_aspect('equal')
    pts = integrate.sample(5000)
    plt.scatter(pts[:, 0], pts[:, 1], s=1, color='k')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if FLAGS.flat:
        plt.savefig('plots/gauss_flat_final.png')
    else:
        plt.savefig('plots/gauss_iflow_final.png')
    plt.close()

if __name__ == '__main__':
    app.run(main)
