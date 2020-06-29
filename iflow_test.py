""" Integrate test functions of the i-flow paper [1]

    [1] i-flow: High-dimensional Integration
        and Sampling with Normalizing Flows
    by: Christina Gao, Joshua Isaacson, and Claudius Krause
    arXiv: 2001.05486
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import erf
import vegas, gvar

from absl import app, flags

from iflow.integration import integrator
from iflow.integration import couplings

tfd = tfp.distributions  # pylint: disable=invalid-name
tfb = tfp.bijectors  # pylint: disable=invalid-name
tf.keras.backend.set_floatx('float64')

FLAGS = flags.FLAGS
flags.DEFINE_string('function', 'Gauss', 'The function to integrate',
                    short_name='f')
flags.DEFINE_integer('ndims', 4, 'The number of dimensions for the integral',
                     short_name='d')
flags.DEFINE_float('alpha', 0.2, 'The width of the Gaussians',
                   short_name='a')
flags.DEFINE_float('radius1', 0.45, 'The outer radius for the annulus integrand',
                   short_name='r1')
flags.DEFINE_float('radius2', 0.2, 'The inner radius for the annulus integrand',
                   short_name='r2')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('ptspepoch', 5000, 'Number of points to sample per epoch',
                     short_name='p')
flags.DEFINE_float('precision', 1e-5, 'Target precision in integrator comparison',
                   short_name='tp')
flags.DEFINE_bool('targetmode', False, 'Flag to trigger training until target precision is reached',
                  short_name='t')

class TestFunctions:
    """ Contains the functions discussed in the reference above.

    Attributes:
        ndims (int): dimensionality of the function to be integrated
        alpha (float): width of the Gaussians in the test functions
                       gauss and camel
        kwargs: additional parameters for the functions (not used at the moment)

    """
    def __init__(self, ndims, alpha, **kwargs):
        self.ndims = ndims
        self.alpha = alpha
        self.variables = kwargs
        self.calls = 0

    def gauss(self, x):
        """ Based on eq. 15 of [1], Gaussian function.

        Integral equals erf(1/(2*alpha)) ** ndims

        Args:
            x (tf.Tensor): Tensor with batch of points to evaluate

        Returns: tf.Tensor: functional values at the given points

        """
        pre = tf.cast(1.0/(self.alpha * tf.sqrt(np.pi))**self.ndims,
                      dtype=tf.float64)
        exponent = -1.0*tf.reduce_sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        self.calls += 1
        return pre * tf.exp(exponent)

    def camel(self, x):
        """ Based on eq. 17 of [1], Camel function.

        The Camel function consists of two Gaussians, centered at
        (1/3, 2/3) in each dimension.

        Integral equals
            (0.5*(erf(1/(3*alpha)) + erf(2/(3*alpha)) ))** ndims

        Args:
            x (tf.Tensor): Tensor with batch of points to evaluate

        Returns: tf.Tensor: functional values at the given points

        """
        pre = tf.cast(1./(self.alpha*tf.sqrt(np.pi))**self.ndims,
                      dtype=tf.float64)
        exponent1 = -1.*tf.reduce_sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = -1.*tf.reduce_sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        self.calls += 1
        return 0.5*pre*(tf.exp(exponent1)+tf.exp(exponent2))

    def circle(self, x):
        """ Based on eq. 19 of [1], two overlapping circles.

        Thickness and height change along the circles.

        Integral equals 0.0136848(1)

        Args:
            x (tf.Tensor): Tensor with batch of points to evaluate

        Returns:
            tf.Tensor: functional values at the given points.

        Raises:
            ValueError: If ndims is not equal to 2.

        """
        if self.ndims != 2:
            raise ValueError("ndims must be equal to 2 for circle function!")
        dx1, dy1, rr, w1, ee = 0.4, 0.6, 0.25, 1./0.004, 3.0
        res = (x[..., 1]**ee
               * tf.exp(-w1*tf.abs((x[..., 1]-dy1)**2
                                   + (x[..., 0]-dx1)**2-rr**2))
               + ((1.0-x[..., 1])**ee)
               * tf.exp(-w1*tf.abs((x[..., 1]-1.0+dy1)**2
                                   + (x[..., 0]-1.0+dx1)**2-rr**2)))
        self.calls += 1
        return res

    def circle_np(self, x):
        """ Based on eq. 19 of [1], two overlapping circles, using numpy.

        Thickness and height change along the circles.

        Integral equals 0.0136848(1)

        Args:
            x (np.array): Array with batch of points to evaluate

        Returns:
            np.array: functional values at the given points.

        Raises:
            ValueError: If ndims is not equal to 2.

        """
        if self.ndims != 2:
            raise ValueError("ndims must be equal to 2 for circle function!")
        dx1, dy1, rr, w1, ee = 0.4, 0.6, 0.25, 1./0.004, 3.0
        res = (x[..., 1]**ee
               * np.exp(-w1*np.abs((x[..., 1]-dy1)**2
                                   + (x[..., 0]-dx1)**2-rr**2))
               + ((1.0-x[..., 1])**ee)
               * np.exp(-w1*np.abs((x[..., 1]-1.0+dy1)**2
                                   + (x[..., 0]-1.0+dx1)**2-rr**2)))
        self.calls += 1
        return res

    def polynom(self, x):
        """ Based on eq. 22 of [1], a polynomial

        Integral equals ndims/6.

        Args:
            x (tf.Tensor): Tensor with batch of points to evaluate

        Returns:
            tf.Tensor: functional values at the given points.
        """

        res = tf.reduce_sum(-x**2 + x, axis=-1)
        self.calls += 1
        return res

    def polynom_np(self, x):
        """ Based on eq. 22 of [1], a polynomial, using numpy

        Integral equals ndims/6.

        Args:
            x (np.array): Array with batch of points to evaluate

        Returns:
            np.array: functional values at the given points.
        """
        res = np.sum(-x**2 + x, axis=-1)
        self.calls += 1
        return res

    class Ring:
        """ Class to store the annulus (ring) function.

        Attributes:
            radius1 (float): Outer radius of the annulus.
            radius2 (float): Inner radius of the annulus.

        """
        def __init__(self, radius1, radius2):
            """ Init annulus function. """

            # Ensure raidus1 is the large one
            if radius1 < radius2:
                radius1, radius2 = radius2, radius1

            self.radius12 = radius1**2
            self.radius22 = radius2**2
            self.calls = 0

        def __call__(self, pts):
            """ Calculate annulus function.

            Args:
                x (tf.Tensor): Tensor with batch of points to evaluate

            Returns:
                tf.Tensor: 1. if on annulus, 0. otherwise

            """
            radius = tf.reduce_sum((pts-0.5)**2, axis=-1)
            out_of_bounds = (radius < self.radius22) | (radius > self.radius12)
            ret = tf.where(out_of_bounds, tf.zeros_like(radius),
                           tf.ones_like(radius))
            self.calls += 1
            return ret

        @property
        def area(self):
            """ Get the area of annulus surface. """
            return np.pi*(self.radius12 - self.radius22)

    class TriangleIntegral:
        """ Class implementing the scalar one-loop triangle. """

        def __init__(self, mass_ext=None, mass_int=None):
            if len(mass_ext) != 3:
                raise ValueError('Triangle requires 3 external masses')
            if len(mass_int) != 3:
                raise ValueError('Triangle requires 3 external masses')
            self.mass_ext = np.array(mass_ext)**2
            self.mass_int = np.array(mass_int)**2
            self.calls = 0

        def FTri(self, t1, t2, perm):
            """ Helper function to evaluate the triangle. """
            return (- self.mass_ext[perm[0]]*t1
                    - self.mass_ext[perm[1]]*t1*t2
                    - self.mass_ext[perm[2]]*t2
                    + (1 + t1 + t2)
                    * (t1*self.mass_int[perm[0]]
                       + t2*self.mass_int[perm[1]]
                       + self.mass_int[perm[2]]))

        def __call__(self, x):
            """ Calculate the one-loop triangle.

            Args:
                x (tf.Tensor): Numpy array with batch of points to evaluate

            Returns: np.ndarray: functional values the given points

            """
            numerator = (1 + x[..., 0] + x[..., 1])**-1.0
            denominator1 = self.FTri(x[..., 0], x[..., 1], [1, 2, 0])
            denominator2 = self.FTri(x[..., 0], x[..., 1], [2, 0, 1])
            denominator3 = self.FTri(x[..., 0], x[..., 1], [0, 1, 2])

            self.calls += 1
            return (- numerator/denominator1
                    - numerator/denominator2
                    - numerator/denominator3)

    class BoxIntegral:
        """ Class implementing the scalar one-loop box. """

        def __init__(self, s12, s23, mass_ext=None, mass_int=None):
            if len(mass_ext) != 4:
                raise ValueError('Box requires 4 external masses')
            if len(mass_int) != 4:
                raise ValueError('Box requires 4 external masses')
            self.mass_ext = np.array(mass_ext)**2
            self.mass_int = np.array(mass_int)**2
            self.s12 = s12
            self.s23 = s23
            self.calls = 0

        def FBox(self, s12, s23, t1, t2, t3, perm):
            """ Helper function to evaluate the box. """
            return (-s12*t2
                    - s23*t1*t3
                    - self.mass_ext[perm[0]]*t1
                    - self.mass_ext[perm[1]]*t1*t2
                    - self.mass_ext[perm[2]]*t2*t3
                    - self.mass_ext[perm[3]]*t3
                    + (1+t1+t2+t3)
                    * (t1*self.mass_int[perm[0]]+t2*self.mass_int[perm[1]]
                       + t3*self.mass_int[perm[2]]+self.mass_int[perm[3]]))

        def __call__(self, x):
            """ Calculate the one-loop box.

            Args:
                x (tf.Tensor): Numpy array with batch of points to evaluate

            Returns: np.ndarray: functional values the given points

            """
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

            self.calls += 1
            return (1.0/denominator1**2
                    + 1.0/denominator2**2
                    + 1.0/denominator3**2
                    + 1.0/denominator4**2)


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
    integrate = integrator.Integrator(func, dist, optimizer,
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
    means = np.zeros(epochs)
    stddevs = np.zeros(epochs)
    for epoch in range(epochs):
        loss, integral, error = integrate.train_one_step(ptspepoch,
                                                         integral=True)
        means[epoch] = integral
        stddevs[epoch] = error
        _, current_precision = variance_weighted_result(means[:epoch+1], stddevs[:epoch+1])
        if epoch % 10 == 0:
            print('Epoch: {:3d} Loss = {:8e} Integral = '
                  '{:8e} +/- {:8e} Total uncertainty = {:8e}'.format(epoch, loss,
                                                                     integral, error,
                                                                     current_precision))

    return means, stddevs

def train_iflow_target(integrate, ptspepoch, target):
    """ Run the iflow integrator

    Args:
        integrate (Integrator): iflow Integrator class object
        ptspepoch (int): number of points per epoch in training
        target (float): target precision of final integral

    Returns:
        numpy.ndarray(float): integral estimations and its uncertainty of each epoch

    """
    means = []
    stddevs = []
    current_precision = 1e99
    epoch = 0
    while current_precision > target:
        loss, integral, error = integrate.train_one_step(ptspepoch,
                                                         integral=True)
        means.append(integral)
        stddevs.append(error)
        _, current_precision = variance_weighted_result(np.array(means), np.array(stddevs))
        if epoch % 10 == 0:
            print('Epoch: {:3d} Loss = {:8e} Integral = '
                  '{:8e} +/- {:8e} Total uncertainty = {:8e}'.format(epoch, loss,
                                                                     integral, error,
                                                                     current_precision))
        epoch += 1
    return np.array(means), np.array(stddevs)

def sample_iflow(integrate, ptspepoch, epochs):
    """ Sample from the iflow integrator

    Args:
        integrate (Integrator): iflow Integrator class object
        ptspepoch (int): number of points per epoch in training
        epochs (int): number of epochs for training

    Returns:
        (tuple): mean and stddev numpy arrays

    """
    # defining a reduced number of epochs for integral evaluation
    red_epochs = int(epochs/5)

    # mean and stddev of trained NF
    print('Estimating integral from trained network')
    means_t = []
    stddevs_t = []
    for _ in range(red_epochs+1):
        mean, var = integrate.integrate(ptspepoch)
        means_t.append(mean)
        stddevs_t.append(tf.sqrt(var/(ptspepoch-1.)).numpy())
    return np.array(means_t), np.array(stddevs_t)

def rel_unc(mean_a, unc_a, mean_b, unc_b):
    """  Relative uncertainty, for Table III """
    ret = mean_a - mean_b
    sqr = np.sqrt(unc_a**2 + unc_b**2)
    ret = ret/sqr
    return ret

def variance_weighted_result(means, stddevs):
    """ Computes weighted mean and stddev of given means and
        stddevs arrays, using Inverse-variance weighting
    """
    assert np.size(means) == np.size(stddevs)
    assert means.shape == stddevs.shape
    variance = 1./np.sum(1./stddevs**2, axis=-1)
    mean = np.sum(means/(stddevs**2), axis=-1)
    mean *= variance
    return mean, np.sqrt(variance)

def variance_unweight(means_wgt, stddevs_wgt):
    """ Computes mean and stddev of individual run given
        means and stddevs arrays that are weighted with
        Inverse-variance weighting up to that run
    """
    inv_var = 1./stddevs_wgt[1:]**2 - 1./stddevs_wgt[:-1]**2
    stddevs = 1./np.sqrt(inv_var)
    stddevs = np.insert(stddevs, 0, stddevs_wgt[0])
    means = means_wgt[1:]/(stddevs_wgt[1:]**2) - means_wgt[:-1]/(stddevs_wgt[:-1]**2)
    means *= stddevs[1:]**2
    means = np.insert(means, 0, means_wgt[0])
    return means, stddevs

def main(argv):
    """ Main function for test runs. """
    del argv

    # gauss: 2, 4, 8, or 16
    # camel: 2, 4, 8, or 16
    # circle: 2
    # annulus: 2
    # Box: ndims = 3, Triangle: ndims = 2
    # Poly: 18, 54, or 96
    ndims = FLAGS.ndims
    alpha = FLAGS.alpha

    plot_FOAM = False

    func = TestFunctions(ndims, alpha)

    # select function:
    if FLAGS.function == 'Gauss':
        target = erf(1/(2.*alpha))**ndims
        integrand = func.gauss
        integrand_np = func.gauss
    elif FLAGS.function == 'Camel':
        target = (0.5*(erf(1/(3.*alpha))+erf(2/(3.*alpha))))**ndims
        integrand = func.camel
        integrand_np = func.camel
    elif FLAGS.function == 'Circle':
        target = 0.0136848
        integrand = func.circle
        integrand_np = func.circle_np
    elif FLAGS.function == 'Ring':
        func_ring = func.Ring(FLAGS.radius1, FLAGS.radius2)
        target = func_ring.area
        integrand = func_ring
        integrand_np = func_ring
    elif FLAGS.function == 'Triangle':
        target = -1.70721682537767509e-5
        integrand = func.TriangleIntegral([0, 0, 125],
                                          [175, 175, 175])
        integrand_np = func.TriangleIntegral([0, 0, 125],
                                             [175, 175, 175])
    elif FLAGS.function == 'Box':
        target = 1.93696402386819321e-10
        integrand = func.BoxIntegral(130**2, -130**2/2.0,
                                     [0, 0, 0, 125],
                                     [175, 175, 175, 175])
        integrand_np = func.BoxIntegral(130**2, -130**2/2.0,
                                        [0, 0, 0, 125],
                                        [175, 175, 175, 175])
    elif FLAGS.function == 'Poly':
        integrand = func.polynom
        integrand_np = func.polynom_np
        target = (1./6.) * ndims

    print("Target value of the Integral in {:d} dimensions is {:.6e}".format(
        ndims, target))

    npts = 100000
    pts = np.array(np.random.rand(npts, ndims), dtype=np.float64)
    value = integrand(pts)

    format_string = ('Crude MC of {} function in {:d} dimensions based on '
                     '{:d} points: {:.3e} +/- {:.3e}')
    print(format_string.format(FLAGS.function, ndims, npts, np.mean(value),
                               np.std(value)/np.sqrt(npts)))


    epochs = FLAGS.epochs
    ptspepoch = FLAGS.ptspepoch
    target_precision = FLAGS.precision * target

    if plot_FOAM:
        filename = "FOAM_data/Foam_{:d}_{}.txt".format(ndims, FLAGS.function)
        print("Reading in FOAM results of {}".format(filename))
        foam_numcalls = []
        foam_means = []
        foam_stddevs = []
        foam_results = []

        one_cell_value = integrand(np.random.rand(ptspepoch, ndims))
        foam_numcalls.append(ptspepoch)
        foam_means.append(np.mean(one_cell_value))
        foam_stddevs.append(np.std(one_cell_value)/np.sqrt(ptspepoch))
        foam_results.append(gvar.gvar("{} +- {}".format(foam_means[-1], foam_stddevs[-1])))

        try:
            foam_file = open(filename, "r")
        except:
            raise NameError("{} not found!".format(filename))
        num_cells = 1
        for line in foam_file:
            splitted_line = line.split(";")
            # check if it is a proper line:
            try:
                splitted_line[1]
            except:
                continue
            foam_results.append(gvar.gvar(splitted_line[0]))
            foam_means.append(foam_results[-1].mean)
            foam_stddevs.append(foam_results[-1].sdev)
            foam_numcalls.append(np.int(splitted_line[1]))
            if foam_stddevs[-1] < target_precision:
                break
            if splitted_line[2] == num_cells:
                break
            num_cells = splitted_line[2]
        foam_file.close()
        foam_calls = np.array(foam_numcalls)
        foam_means = np.array(foam_means)
        foam_stddevs = np.array(foam_stddevs)
        foam_results = np.array(foam_results)
        print("Done")
        print("Weighted FOAM result is {:.5e} +/- {:.5e}".format(
            foam_means[-1], foam_stddevs[-1]))
        print("Relative Uncertainty FOAM result is {:.3f}".format(
            rel_unc(foam_means[-1], foam_stddevs[-1], target, 0.)))
        print("FOAM used {} cells and {:d} function calls".format(int(num_cells)+1, foam_numcalls[-1]))

    if not FLAGS.targetmode:
        # epoch mode
        x_values = np.arange(ptspepoch, (epochs + 1) * ptspepoch, ptspepoch)

        # i-flow
        integrate = build_iflow(integrand, ndims)
        mean_t, err_t = train_iflow(integrate, ptspepoch, epochs)

        iflow_mean_wgt, iflow_err_wgt = variance_weighted_result(mean_t, err_t)

        print("Results for {:d} dimensions:".format(ndims))
        print("Weighted iflow result is {:.5e} +/- {:.5e}".format(
            iflow_mean_wgt, iflow_err_wgt))

        print("Relative Uncertainty iflow result is {:.3f}".format(
            rel_unc(iflow_mean_wgt, iflow_err_wgt, target, 0.)))

        # vegas
        vegas_integ = vegas.Integrator(ndims* [[0, 1]])
        if FLAGS.function in ['Box', 'Ring', 'Triangle']:
            current_vegas_calls = integrand_np.calls
        else:
            current_vegas_calls = func.calls

        vegas_calls = []
        vegas_results = []
        vegas_means = []
        vegas_stddevs = []
        for i in range(epochs):
            current_result = vegas_integ(integrand_np, nitn=1, neval=ptspepoch)
            vegas_means.append(current_result.mean)
            vegas_stddevs.append(current_result.sdev)
            vegas_results.append(current_result)

            if FLAGS.function in ['Box', 'Ring', 'Triangle']:
                vegas_calls.append(integrand_np.calls - current_vegas_calls)
                current_vegas_calls = integrand_np.calls
            else:
                vegas_calls.append(func.calls - current_vegas_calls)
                current_vegas_calls = func.calls

            _, current_precision = variance_weighted_result(np.array(vegas_means),
                                                            np.array(vegas_stddevs))
            if i % 10 == 0:
                print('Epoch: {:3d} Integral = '
                      '{:8e} +/- {:8e} Total uncertainty = {:8e}'.format(i, current_result.mean,
                                                                         current_result.sdev,
                                                                         current_precision))
        vegas_calls = np.array(vegas_calls)
        vegas_means = np.array(vegas_means)
        vegas_stddevs = np.array(vegas_stddevs)

        vegas_mean_wgt, vegas_err_wgt = variance_weighted_result(vegas_means, vegas_stddevs)
        print("Weighted VEGAS result is {:.5e} +/- {:.5e}".format(
            vegas_mean_wgt, vegas_err_wgt))
        print("Relative Uncertainty VEGAS result is {:.3f}".format(
            rel_unc(vegas_mean_wgt, vegas_err_wgt, target, 0.)))


        plt.figure(dpi=150, figsize=[5., 4.])
        #plt.xlim(ptspepoch, np.maximum(epochs * ptspepoch, np.sum(vegas_calls)))
        plt.xlim(ptspepoch, epochs * ptspepoch)
        plt.xlabel('Evaluations in training')
        plt.ylim(1e-5, 1e1)
        plt.ylabel('Integral uncertainty (%)')
        plt.yscale('log')
        plt.xscale('log')

        # Plot Integral Uncertainty
        plt.plot(x_values, err_t/np.abs(mean_t), color='r', label='i-flow')
        plt.plot(np.cumsum(vegas_calls), vegas_stddevs/np.abs(vegas_means), color='b',
                 label='VEGAS')
        if plot_FOAM:
            foam_means_uwgt, foam_stddevs_uwgt = variance_unweight(foam_means, foam_stddevs)
            plt.plot(foam_calls, foam_stddevs_uwgt/np.abs(foam_means_uwgt), color='green',
                     label='FOAM')
        plt.legend()

        plt.savefig('{}_unc.png'.format(FLAGS.function), bbox_inches='tight')
        plt.show()
        plt.close()

        # plot total precision
        plt.figure(dpi=150, figsize=[5., 4.])
        plt.yscale('log')
        plt.xscale('log')
        #plt.xlim(ptspepoch, np.maximum(epochs * ptspepoch, np.sum(vegas_calls)))
        plt.xlim(ptspepoch, epochs * ptspepoch)
        plt.xlabel('Evaluations in training')
        plt.ylim(target_precision/(2.*target), 1e0)
        plt.ylabel('Total relative integral uncertainty')

        # Plot iflow
        total_uncertainty = np.zeros(epochs)
        for i in range(epochs):
            #print(variance_weighted_result(mean_t[:i+1], err_t[:i+1]))
            _, total_uncertainty[i] = variance_weighted_result(mean_t[:i+1], err_t[:i+1])
        plt.plot(x_values, total_uncertainty/target, color='r', label='i-flow')

        # plot VEGAS
        len_vegas = len(vegas_calls)
        vegas_total_uncertainty = np.zeros(len_vegas)
        for i in range(len_vegas):
            _, vegas_total_uncertainty[i] = variance_weighted_result(vegas_means[:i+1],
                                                                     vegas_stddevs[:i+1])
        plt.plot(np.cumsum(vegas_calls), vegas_total_uncertainty/target, color='b', label='VEGAS')
        if plot_FOAM:
            plt.plot(foam_calls, foam_stddevs/target, color='green', label='FOAM')

        # background grid
        start_values = np.logspace(-3, 4, 8)
        for start in start_values:
            plt.plot(x_values, start/np.sqrt(x_values), color='gray', lw=0.5, ls='dashed')

        plt.legend()
        plt.savefig('{}_{:d}_total_unc.png'.format(FLAGS.function, ndims), bbox_inches='tight')
        plt.show()
        plt.close()

    else:
        # target mode
        print("In target mode with absolute precision {}, based on relative precision {}".format(
            target_precision, FLAGS.precision))
        integrate = build_iflow(integrand, ndims)
        mean_t, err_t = train_iflow_target(integrate, ptspepoch, target_precision)
        num_epochs = len(mean_t)
        x_values = np.arange(ptspepoch, (num_epochs+1) * ptspepoch, ptspepoch)
        iflow_mean_wgt, iflow_err_wgt = variance_weighted_result(mean_t, err_t)

        print("Results for {:d} dimensions:".format(ndims))
        print("Weighted iflow result is {:.5e} +/- {:.5e}.".format(
            iflow_mean_wgt, iflow_err_wgt))
        print("Relative Uncertainty iflow result is {:.3f}".format(
            rel_unc(iflow_mean_wgt, iflow_err_wgt, target, 0.)))
        print("i-flow needed {:d} epochs and {:d} function calls".format(num_epochs,
                                                                         num_epochs*ptspepoch))

        # vegas
        vegas_integ = vegas.Integrator(ndims* [[0, 1]])
        if FLAGS.function in ['Box', 'Ring', 'Triangle']:
            current_vegas_calls = integrand_np.calls
        else:
            current_vegas_calls = func.calls
        vegas_calls = []
        vegas_results = []
        vegas_means = []
        vegas_stddevs = []

        first_run = True

        epoch = 0
        current_vegas_precision = 1e99
        while current_vegas_precision > target_precision:
            current_result = vegas_integ(integrand_np, nitn=1, neval=ptspepoch)
            vegas_means.append(current_result.mean)
            vegas_stddevs.append(current_result.sdev)
            vegas_results.append(current_result)
            if FLAGS.function in ['Box', 'Ring', 'Triangle']:
                vegas_calls.append(integrand_np.calls - current_vegas_calls)
                current_vegas_calls = integrand_np.calls
            else:
                vegas_calls.append(func.calls - current_vegas_calls)
                current_vegas_calls = func.calls

            _, current_vegas_precision = variance_weighted_result(np.array(vegas_means),
                                                                  np.array(vegas_stddevs))
            if epoch % 10 == 0:
                print('Epoch: {:3d} Integral = '
                      '{:8e} +/- {:8e} Total uncertainty = {:8e}'.format(epoch, current_result.mean,
                                                                         current_result.sdev,
                                                                         current_vegas_precision))
            epoch += 1
            if np.sum(vegas_calls) > 50000000:
                break
        vegas_calls = np.array(vegas_calls)
        vegas_means = np.array(vegas_means)
        vegas_stddevs = np.array(vegas_stddevs)

        vegas_mean_wgt, vegas_err_wgt = variance_weighted_result(vegas_means, vegas_stddevs)
        print("Weighted VEGAS result is {:.5e} +/- {:.5e}".format(
            vegas_mean_wgt, vegas_err_wgt))
        print("Relative Uncertainty VEGAS result is {:.3f}".format(
            rel_unc(vegas_mean_wgt, vegas_err_wgt, target, 0.)))
        print("VEGAS needed {:d} iterations and {:d} function calls".format(epoch,
                                                                            np.sum(vegas_calls)))

        # plot relative integral uncertainty per epoch
        plt.figure(dpi=150, figsize=[5., 4.])
        #plt.xlim(ptspepoch, np.maximum(epochs * ptspepoch, np.sum(vegas_calls)))
        plt.xlim(ptspepoch, num_epochs * ptspepoch)
        plt.xlabel('Evaluations in training')
        plt.ylim(1e-5, 1e1)
        plt.ylabel('Integral uncertainty (%)')
        plt.yscale('log')
        plt.xscale('log')

        # Plot iflow
        plt.plot(x_values, err_t/np.abs(mean_t), color='r', label='i-flow')
        plt.plot(np.cumsum(vegas_calls), vegas_stddevs/np.abs(vegas_means), color='b',
                 label='VEGAS')
        if plot_FOAM:
            foam_means_uwgt, foam_stddevs_uwgt = variance_unweight(foam_means, foam_stddevs)
            plt.plot(foam_calls, foam_stddevs_uwgt/np.abs(foam_means_uwgt), color='green',
                     label='FOAM')
        plt.legend()

        plt.savefig('{}_{:d}_unc_target.png'.format(FLAGS.function, ndims), bbox_inches='tight')
        plt.show()
        plt.close()

        # plot total precision
        plt.figure(dpi=150, figsize=[5., 4.])
        plt.yscale('log')
        plt.xscale('log')
        #plt.xlim(ptspepoch, np.maximum(epochs * ptspepoch, np.sum(vegas_calls)))
        plt.xlim(ptspepoch, num_epochs * ptspepoch)
        plt.xlabel('Evaluations in training')
        plt.ylim(target_precision/(2.*target), 1e0)
        plt.ylabel('Total relative integral uncertainty')

        # Plot iflow
        total_uncertainty = np.zeros(num_epochs)
        for i in range(num_epochs):
            #print(variance_weighted_result(mean_t[:i+1], err_t[:i+1]))
            _, total_uncertainty[i] = variance_weighted_result(mean_t[:i+1], err_t[:i+1])
        plt.plot(x_values, total_uncertainty/target, color='r', label='i-flow')

        # plot VEGAS
        len_vegas = len(vegas_calls)
        vegas_total_uncertainty = np.zeros(len_vegas)
        for i in range(len_vegas):
            _, vegas_total_uncertainty[i] = variance_weighted_result(vegas_means[:i+1],
                                                                     vegas_stddevs[:i+1])
        plt.plot(np.cumsum(vegas_calls), vegas_total_uncertainty/target, color='b', label='VEGAS')
        if plot_FOAM:
            plt.plot(foam_calls, foam_stddevs/target, color='green', label='FOAM')

        # background grid
        start_values = np.logspace(-3, 4, 8)
        for start in start_values:
            plt.plot(x_values, start/np.sqrt(x_values), color='gray', lw=0.5, ls='dashed')

        plt.legend()
        plt.savefig('{}_{:d}_total_unc_target.png'.format(FLAGS.function, ndims), bbox_inches='tight')
        plt.show()
        plt.close()


    # scatter plot for ring
    if FLAGS.function == 'Ring':
        pts = integrate.sample(7500)
        fig = plt.figure(dpi=150, figsize=[4., 4.])
        axis = fig.add_subplot(111)
        radius2 = (pts[:, 0]-0.5)**2 + (pts[:, 1]-0.5)**2
        in_ring = np.logical_and(radius2 < func_ring.radius12,
                                 radius2 > func_ring.radius22)
        _, pts_inout = np.unique(in_ring, return_counts=True)
        print('{:d} points were generated, {:d} of them are on the ring, {:d} '
              'are outside.'.format(np.sum(pts_inout),
                                    pts_inout[1],
                                    pts_inout[0]))
        print('Cut efficiency: {:.4f}'.format(pts_inout[1]/np.sum(pts_inout)))
        inner = np.sqrt(func_ring.radius12)
        outer = np.sqrt(func_ring.radius22)
        color_ring = np.where(in_ring, 'blue', 'red')
        inner_circle = plt.Circle((0.5, 0.5), inner, color='k', fill=False)
        outer_circle = plt.Circle((0.5, 0.5), outer, color='k', fill=False)
        plt.scatter(pts[:, 0], pts[:, 1], s=1, color=color_ring)
        axis.add_artist(inner_circle)
        axis.add_artist(outer_circle)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig('ring_scatter.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    app.run(main)
