""" Integrate test functions of the original VEGAS paper [1]
    and compare them to i-flow

    [1] https://doi.org/10.1016/0021-9991(78)90004-9
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from absl import app
import vegas

from flow.integration import integrator
from flow.integration import couplings

tfd = tfp.distributions  # pylint: disable=invalid-name
tfb = tfp.bijectors  # pylint: disable=invalid-name
tf.keras.backend.set_floatx('float64')


class TestFunctions:
    """ Contains the functions discussed in the reference above """

    def __init__(self, ndims, alpha, **kwargs):
        self.ndims = ndims
        self.alpha = alpha
        self.variables = kwargs

    def gauss(self, x):
        """ Based on eq. 8 of [1]
            Integral equals 1.

        Args:
            x (np.ndarray): Numpy array with batch of points to evaluate

        Returns: np.ndarray: functional values the given points

        """
        pre = 1./(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.*np.sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        return pre * np.exp(exponent)

    def camel(self, x):
        """ Based on eq. 9 of [1]
            Integral equals 1.

        Args:
            x (np.ndarray): Numpy array with batch of points to evaluate

        Returns: np.ndarray: functional values the given points

        """
        pre = 1./(self.alpha*np.sqrt(np.pi))**self.ndims
        exponent1 = -1.*np.sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = -1.*np.sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        return 0.5*pre*(np.exp(exponent1)+np.exp(exponent2))

    def camel_tf(self, x):
        """ Based on eq. 9 of [1]
            Integral equals 1.

        Args:
            x (tf.ndarray): Numpy array with batch of points to evaluate

        Returns: tf.ndarray: functional values the given points

        """
        pre = tf.cast(1./(self.alpha*tf.sqrt(np.pi))**self.ndims, dtype=tf.float64)
        exponent1 = -1.*tf.reduce_sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = -1.*tf.reduce_sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        return 0.5*pre*(tf.exp(exponent1)+tf.exp(exponent2))

    def tsuda(self, x):
        """ Based on eq. 10 of [1]
            Integral equals 1.

        Args:
            x (np.ndarray): Numpy array with batch of points to evaluate

        Returns: np.ndarray: functional values the given points

        """
        if self.ndims != 8:
            raise "ndims must be equal to 8 for Tsuda function!"
        c = 1./(np.sqrt(10)-1.)
        ret = np.prod((c/(c+1)) * ((c+1)/(c+x))**2, axis=-1)
        return ret

    def line(self, x):
        """ Based on test function of S. Hoeche's Foam
            Integral equals ???

        Args:
            x (np.ndarray): Numpy array giving the point to evaluate

        Returns: float64: value of the line function

        """
        w1 = 1./0.004
        return np.exp(-w1*(np.sum(x, axis=-1)-1.0)**2)

    def circle(self, x):
        """ Based on test function of S. Hoeche's Foam
            Integral equals ???

        Args:
            x (np.ndarray): Numpy array with batch of points to evaluate

        Returns: np.ndarray: functional values the given points

        """
        if self.ndims != 2:
            raise "ndims must be equal to 2 for circle function!"
        dx1, dy1, rr, w1, ee = 0.4, 0.6, 0.25, 1./0.004, 3.0
        res = (np.power(x[..., 1], ee)
               * np.exp(-w1*np.abs((x[..., 1]-dy1)**2
                                   + (x[..., 0]-dx1)**2-rr**2))
               + np.power(1.0-x[..., 1], ee)
               * np.exp(-w1*np.abs((x[..., 1]-1.0+dy1)**2
                                   + (x[..., 0]-1.0+dx1)**2-rr**2)))
        return res

    def box_integral(self, x):
        def dot2(mom):
            return mom[..., 0]**2 - (mom[..., 1]**2 
                                     + mom[..., 2]**2 
                                     + mom[..., 3]**2)

        if self.ndims != 4:
            raise "ndims must be equal to 6 for the box integral!"
        energy = np.sqrt(self.variables['s'])/2.0
        theta = self.variables['angle']
        mom1 = np.array([energy, 0, 0, energy])
        mom2 = np.array([energy, 0, 0, -energy])
        mom3 = np.array([energy, energy*np.sin(theta),
                         0, energy*np.cos(theta)])
        mom4 = np.array([energy, -energy*np.sin(theta),
                         0, -energy*np.cos(theta)])

        mag = x[..., 0]/(1 - x[..., 0])
        theta = 2*np.pi*x[..., 1]
        cphi = 2*x[..., 2] - 1.0
        sphi = np.sqrt(1 - cphi**2)
        comega = 2*x[..., 3] - 1.0
        somega = np.sqrt(1 - comega**2)
        dps = 8*np.pi/(1 - x[..., 0])**2
        loop_mom = np.array([mag*np.cos(theta),
                             mag*np.sin(theta)*sphi*comega,
                             mag*np.sin(theta)*sphi*somega,
                             mag*np.sin(theta)*cphi]).T

        denominator = (dot2(loop_mom)*dot2(loop_mom+mom1)
                       * dot2(loop_mom+mom1+mom2)*dot2(loop_mom+mom4))
        return dps/denominator


def run_vegas(func, ndims, ptspepoch, epochs):
    """ running VEGAS for integral + stddev vs. training

    Args:
    func:            integrand
    ndims (int):     dimensionality of the integrand
    ptspepoch (int): number of points per epoch in training
    epochs (int):    number of epochs for training
    ptsint (int):    number of points used to plot integral and uncertainty
    """

    integ = vegas.Integrator(ndims*[[0., 1.]])

    x_values = []
    means = []
    stddevs = []

    mean, stddev = evaluate(func, integ, ptspepoch)
    print("Result = {:.3f} +/- {:.3f} ".format(mean, stddev))
    means.append(mean)
    stddevs.append(stddev)

    results = integ(func, nitn=epochs, neval=ptspepoch, adapt=True, maxinc_axis=100)  # , max_nhcube = 1)
    print(results.summary())

    for number in results.itn_results:
        means.append(number.mean)
        stddevs.append(number.sdev)
        # print("value = {} +/- {} ".format(value, err))

    # mean, stddev = evaluate(func, integ, ptsint)
    means = np.array(means)
    stddevs = np.array(stddevs)

    # print("Result = {:.3f} +/- {:.3f} ".format(mean, stddev))

    # for epoch in range(epochs):
    #     integ(func, nitn=1, neval=ptspepoch, adapt=True, max_nhcube = 1)
    #     mean, stddev = evaluate(func, integ, ptsint)
    #     x_values.append((epoch+1)*ptspepoch)
    #     means.append(mean)
    #     stddevs.append(stddev)
    #     print("Epoch {:d} of {:d} done".format(epoch+1, epochs))
    # x_values = np.array(x_values)
    # means = np.array(means)
    # stddevs = np.array(stddevs)
    return means, stddevs


def build(in_features, out_features, options):
    " Build the NN. """
    del options

    invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
    hidden = tf.keras.layers.Dense(128, activation='relu')(invals)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_features)(hidden)
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
    sub_masks = np.array([binary_list(i, n_masks)
                          for i in range(ndims)]).T[::-1]
    flip_masks = mask_flip(sub_masks)

    # Combine masks
    masks = np.empty((2*n_masks, ndims))
    masks[0::2] = flip_masks
    masks[1::2] = sub_masks

    return masks


def run_iflow(func, ndims, ptspepoch, epochs):
    """ Run the iflow integrator

    Args:
        func: integrand
        ndims (int): dimensionality of the integrand
        ptspepoch (int): number of points per epoch in training
        epochs (int): number of epochs for training

    Returns: (tuple): List of means and standard deviations for each epoch

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
    integrate = integrator.Integrator(func, dist, optimizer,
                                      loss_func='exponential')

    means = np.zeros(epochs+1)
    stddevs = np.zeros(epochs+1)
    for epoch in range(epochs+1):
        loss, integral, error = integrate.train_one_step(ptspepoch,
                                                         integral=True)
        means[epoch] = integral
        stddevs[epoch] = error
        if epoch % 10 == 0:
            print('Epoch: {:3d} Loss = {:8e} Integral = '
                  '{:8e} +/- {:8e}'.format(epoch, loss, integral, error))

    return means, stddevs


def evaluate(func, integ, ptsint=100000):
    """
    Takes an instance of Vegas and evaluates the integral
    without further adaptation

    Args:
        func:   Integrand to be integrated
        integ:  An instance of vegas.Integrator
        ptsint: Number of points used for evaluation
    """
    result = integ(func, nitn=1, neval=ptsint, adapt=False, max_nhcube=1)
    return result.mean, result.sdev


def main(argv):
    # tf.random.set_seed(1236)
    """ Main function for test runs.

    TODO:
    initialize vegas
    run vegas
    initialize i-flow
    run i-flow
    produce plots of target and learned distribution
    """
    del argv
    tf.config.experimental_run_functions_eagerly(True)

    ndims = 4
    alpha = 0.1

    npts = 100000
    pts = np.random.rand(npts, ndims)

    # select function:
    func = TestFunctions(ndims, alpha, s=100, angle=np.pi/2.0)

    value = func.box_integral(pts)

    format_string = ('Crude MC of {} function in {:d} dimensions: '
                     '{:.3f} +/- {:.3f}')
    print(format_string.format('Box', ndims, np.mean(value),
                               np.std(value)/np.sqrt(npts)))

    epochs = 400
    ptspepoch = 5000
    x_values = np.arange(0, (epochs + 1) * ptspepoch, ptspepoch)

    vegas_mean, vegas_err = run_vegas(func.box_integral, ndims, ptspepoch, epochs)
    iflow_mean, iflow_err = run_iflow(func.box_integral, ndims, ptspepoch, epochs)
    plt.figure(dpi=150, figsize=[5., 4.])
    plt.xlim(0., epochs * ptspepoch)
    plt.xlabel('Evaluations in training')
    plt.ylim(0., 2.)
    plt.ylabel('Integral value')
    # plt.yscale('log')

    # Plot Vegas
    plt.plot(x_values, vegas_mean, color='b')
    plt.fill_between(x_values, vegas_mean + vegas_err, vegas_mean - vegas_err,
                     color='b', alpha=0.5)

    # Plot iflow
    plt.plot(x_values, iflow_mean, color='r')
    plt.fill_between(x_values, iflow_mean + iflow_err, iflow_mean - iflow_err,
                     color='r', alpha=0.5)

    # True value
    plt.plot([0., epochs * ptspepoch], [1., 1.], ls='--', color='k')
    # plt.title('VEGAS integral')
    plt.savefig('camel_{}.png'.format(ndims), bbox_inches='tight')
    plt.show()

    plt.figure(dpi=150, figsize=[5., 4.])
    plt.xlim(0., epochs * ptspepoch)
    plt.xlabel('Evaluations in training')
    plt.ylim(1e-4, 1e1)
    plt.ylabel('Integral uncertainty (%)')
    plt.yscale('log')
    # Plot Vegas
    plt.plot(x_values, vegas_err/vegas_mean, color='b')

    # Plot iflow
    plt.plot(x_values, iflow_err/iflow_mean, color='r')

    # plt.title('VEGAS uncertainty')
    plt.savefig('camel_{}_unc.png'.format(ndims), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    app.run(main)
