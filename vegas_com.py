""" Integrate test functions of the original VEGAS paper [1]
    and compare them to i-flow

    [1] https://doi.org/10.1016/0021-9991(78)90004-9
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from absl import app
import vegas

from flow.integration import integrator
from flow.integration import couplings


class TestFunctions:
    """ Contains the functions discussed in the reference above """

    def __init__(self, ndims, alpha):
        self.ndims = ndims
        self.alpha = alpha

    def gauss(self, x):
        """ Based on eq. 8 of [1] """
        pre = 1./(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = - np.sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        return pre * np.exp(exponent)

    def camel(self, x):
        """ Based on eq. 9 of [1] """
        pre = 1./(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent1 = - np.sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = - np.sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        return 0.5*pre * (np.exp(exponent1)+np.exp(exponent2))

    def tsuda(self, x):
        """ Based on eq. 10 of [1] """
        if self.ndims != 8:
            raise "ndims must be equal to 8 for Tsuda function!"
        c = 1./(np.sqrt(10)-1.)
        ret = np.prod((c/(c+1)) * ((c+1)/(c+x))**2, axis=-1)
        return ret

    def line(self, x):
        """ TODO: Definition

        Args:
            x (np.ndarray): Numpy array giving the point to evaluate

        Returns: float64: value of the line function

        """
        pass


def main(argv):
    # tf.random.set_seed(1236)
    """ Main function for test runs. """
    del argv

    ndims = 8
    alpha = .1

    npts = 10000
    test = np.random.rand(npts, ndims)
    func = TestFunctions(ndims, alpha)

    value1 = func.gauss(test)
    value2 = func.camel(test)
    value3 = func.tsuda(test)

    format_string = '{:.3e} +/- {:.3e}'
    print(format_string.format(np.mean(value1), np.std(value1)/np.sqrt(npts)))
    print(format_string.format(np.mean(value2), np.std(value2)/np.sqrt(npts)))
    print(format_string.format(np.mean(value3), np.std(value3)/np.sqrt(npts)))


if __name__ == '__main__':
    app.run(main)
