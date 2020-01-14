""" Integrate test functions of the i-flow paper [1]
    
    [1] i-flow: High-dimensional Integration 
        and Sampling with Normalizing Flows
    by: Christina Gao, Joshua Isaacson, 
    and Claudius Krause
    arXiv: 2001.xxxxx
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from absl import app

# should that be i-flow?
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
        """ Based on eq. 10 of [1]
            Integral equals erf(1/(2*alpha)) ** ndims

        Args:
            x (np.ndarray): Numpy array with batch of points to evaluate

        Returns: np.ndarray: functional values the given points

        """
        pre = 1./(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.*np.sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        return pre * np.exp(exponent)

    def camel(self, x):
        """ Based on eq. 12 of [1]
            Integral equals
            (0.5*(erf(1/(3*alpha)) + erf(2/(3*alpha)) ))** ndims

        Args:
            x (np.ndarray): Numpy array with batch of points to evaluate

        Returns: np.ndarray: functional values the given points

        """
        pre = 1./(self.alpha*np.sqrt(np.pi))**self.ndims
        exponent1 = -1.*np.sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = -1.*np.sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        return 0.5*pre*(np.exp(exponent1)+np.exp(exponent2))


    def circle(self, x):
        """ Based on eq. 14 of [1]
            Integral equals 0.0136848(1) 

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

    class Ring:
        """ Class to store the ring function. """

        def __init__(self, radius1, radius2):
            """ Init ring function. """

            # Ensure raidus1 is the large one
            if radius1 < radius2:
                radius1, radius2 = radius2, radius1

            self.radius12 = radius1**2
            self.radius22 = radius2**2

        def __call__(self, pts):
            """ Calculate a ring like function. """
            radius = tf.reduce_sum((pts-0.5)**2, axis=-1)
            out_of_bounds = (radius < self.radius22) | (radius > self.radius12)
            ret = tf.where(out_of_bounds, tf.zeros_like(radius), tf.ones_like(radius))
            return ret
        @property
        def area(self):
            """ Get the area of ring surface. """
            return np.pi*(self.radius12 - self.radius22)

    class triangle_integral:

        def __init__(self, mass_ext=None, mass_int=None):
            if len(mass_ext) != 3:
                raise ValueError('Triangle requires 3 external masses')
            if len(mass_int) != 3:
                raise ValueError('Triangle requires 3 external masses')
            self.mass_ext = np.array(mass_ext)**2
            self.mass_int = np.array(mass_int)**2

        def FTri(self, t1, t2, perm):
            return (-self.mass_ext[perm[0]]*t1
                    -self.mass_ext[perm[1]]*t1*t2
                    -self.mass_ext[perm[2]]*t2
                    + (1 + t1 + t2)
                    * (t1*self.mass_int[perm[0]]
                       + t2*self.mass_int[perm[1]]
                       + self.mass_int[perm[2]]))

        def __call__(self, x):
            numerator = (1 + x[..., 0] + x[..., 1])**-1.0
            denominator1 = self.FTri(x[..., 0], x[..., 1], [1, 2, 0])
            denominator2 = self.FTri(x[..., 0], x[..., 1], [2, 0, 1])
            denominator3 = self.FTri(x[..., 0], x[..., 1], [0, 1, 2])

            return -numerator/denominator1-numerator/denominator2-numerator/denominator3

    class box_integral:

        def __init__(self, s12, s23, mass_ext=None, mass_int=None):
            if len(mass_ext) != 4:
                raise ValueError('Box requires 4 external masses')
            if len(mass_int) != 4:
                raise ValueError('Box requires 4 external masses')
            self.mass_ext = np.array(mass_ext)**2
            self.mass_int = np.array(mass_int)**2
            self.s12 = s12
            self.s23 = s23

        def FBox(self, s12, s23, t1, t2, t3, perm):
            return (-s12*t2
                    -s23*t1*t3
                    -self.mass_ext[perm[0]]*t1
                    -self.mass_ext[perm[1]]*t1*t2
                    -self.mass_ext[perm[2]]*t2*t3
                    -self.mass_ext[perm[3]]*t3
                    +(1+t1+t2+t3)
                    *(t1*self.mass_int[perm[0]]+t2*self.mass_int[perm[1]]
                      +t3*self.mass_int[perm[2]]+self.mass_int[perm[3]]))

        def __call__(self, x):
            denominator1 = self.FBox(self.s23, self.s12,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [1, 2, 3, 0])
            denominator2 = self.FBox(self.s12, self.s23,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [2, 3, 0, 1])
            denominator3 = self.FBox(self.s23, self.s12,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [3, 0, 1, 2])
            denominator4 = self.FBox(self.s12, self.s23,
                                     x[..., 0], x[..., 1], x[..., 2],
                                     [0, 1, 2, 3])

            return (1.0/denominator1**2
                    +1.0/denominator2**2
                    +1.0/denominator3**2
                    +1.0/denominator4**2)


def build(in_features, out_features, options):
    " Build the NN. """
    del options

    invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
    hidden = tf.keras.layers.Dense(32, activation='relu')(invals)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(32, activation='relu')(hidden)
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

    # defining a reduced number of epochs for integral evaluation 
    red_epochs = int(epochs/5)

    # mean and stddev of trained NF
    means_t = []
    stddevs_t = []
    for epoch in range (red_epochs +1):
        mean, var = integrate.integrate(ptspepoch)
        means_t.append(mean)
        stddevs_t.append(tf.sqrt(var/ptspepoch).numpy())
    return means, stddevs, np.array(means_t), np.array(stddevs_t)


def main(argv):
    """ Main function for test runs.

    TODO:
    clean up
    put flags
    crosscheck
    """
    del argv
    tf.config.experimental_run_functions_eagerly(True)

    # gauss: 2, 4, 8, or 16
    # camel: 2, 4, 8, or 16
    # circle: 2
    # ring: 2
    # Box: ndims = 3, Triangle: ndims = 2
    ndims = 4
    alpha = 0.2

    func = TestFunctions(ndims, alpha, s=100, angle=np.pi/2.0)    
    # select function:
    # ring:
    #func_ring = func.Ring(0.45,0.2)
    
    # triangle:
    # tri_integral = func.triangle_integral([0, 0, 125],
    #                                       [175, 175, 175])
    # box:
    #box_integral = func.box_integral(130**2, -130**2/2.0,
    #                                 [0, 0, 0, 125],
    #                                 [175, 175, 175, 175])

    from scipy.special import erf
    # gauss:
    target = erf(1/(2.*alpha))**ndims
    # camel:
    #target = (0.5*(erf(1/(3.*alpha))+erf(2/(3.*alpha))))**ndims
    # circle:
    #target = 0.0136848
    # ring:
    #target = func_ring.area
    # box:
    #target = 1.936964e-10
    print("Target value of the Integral in {:d} dimensions is {:.6e}".format(ndims ,target))
    
    #value = func_ring(pts)
    npts = 100000
    pts = np.random.rand(npts, ndims)
    value = func.gauss(pts)

    format_string = ('Crude MC of {} function in {:d} dimensions based on {:d} points: '
                     '{:.3e} +/- {:.3e}')
    # print(format_string.format('Triangle', ndims, np.mean(value),
    #                            np.std(value)/np.sqrt(npts)))
    print(format_string.format('Gauss', ndims, npts, np.mean(value),
                               np.std(value)/np.sqrt(npts)))

    epochs = 1000
    ptspepoch = 5000
    x_values = np.arange(0, (epochs + 1) * ptspepoch, ptspepoch)
    
    iflow_mean_t, iflow_err_t, iflow_mean_e, iflow_err_e = run_iflow(func.gauss, ndims, ptspepoch, epochs)

    iflow_mean_wgt = 0.
    iflow_err_wgt = 0.
    for i in range(len(iflow_mean_e)):
        iflow_err_wgt += 1./(iflow_err_e[i]**2)
        iflow_mean_wgt += iflow_mean_e[i]/(iflow_err_e[i]**2)
    iflow_mean_wgt = iflow_mean_wgt / iflow_err_wgt
    iflow_err_wgt = 1. / np.sqrt(iflow_err_wgt)

    print("Results for {:d} dimensions:".format(ndims))
    print("Weighted iflow result is {:.5e} +/- {:.5e}".format(iflow_mean_wgt, iflow_err_wgt))

    # numbers for the table III (relative uncertainty):
    def rel_unc(meanA,uncA,meanB,uncB):
        ret = meanA - meanB
        sqr = np.sqrt(uncA**2 + uncB**2)
        ret = ret/sqr
        return ret
        
    print("Weighted iflow result is {:.3f}".format(rel_unc(iflow_mean_wgt, iflow_err_wgt, target, 0.)))

    plt.figure(dpi=150, figsize=[5., 4.])
    plt.xlim(0., epochs * ptspepoch)
    plt.xlabel('Evaluations in training')
    plt.ylim(1e-5, 1e1)
    plt.ylabel('Integral uncertainty (%)')
    plt.yscale('log')
    # Plot iflow
    plt.plot(x_values, iflow_err_t/np.abs(iflow_mean_t), color='r')

    plt.savefig('Gauss_unc.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    app.run(main)
