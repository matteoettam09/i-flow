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
        pre = 1./(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent1 = -1. * np.sum(((x-(1./3.))**2)/self.alpha**2, axis=-1)
        exponent2 = -1. * np.sum(((x-(2./3.))**2)/self.alpha**2, axis=-1)
        return 0.5*pre * (np.exp(exponent1)+np.exp(exponent2))

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
        res = np.power(x[...,1],ee)*np.exp(-w1*np.abs((x[...,1]-dy1)**2+(x[...,0]-dx1)**2-rr**2))+\
            np.power(1.0-x[...,1],ee)*np.exp(-w1*np.abs((x[...,1]-1.0+dy1)**2+(x[...,0]-1.0+dx1)**2-rr**2))
        return res


def main(argv):
    # tf.random.set_seed(1236)
    """ Main function for test runs. """

    """ TODO:
    initialize vegas
    run vegas
    initialize i-flow
    run i-flow
    produce plots of target and learned distribution
    """
    del argv

    ndims1 = 8
    ndims2 = 2
    alpha = .1

    npts = 10000
    pts1 = np.random.rand(npts, ndims1)
    pts2 = np.random.rand(npts, ndims2)
    
    func1 = TestFunctions(ndims1, alpha)
    func2 = TestFunctions(ndims2, alpha)

    value1 = func1.gauss(pts1)
    value2 = func1.camel(pts1)
    value3 = func1.tsuda(pts1)
    value4 = func1.line(pts1)
    value5 = func2.gauss(pts2)
    value6 = func2.camel(pts2)
    value7 = func2.circle(pts2)
    value8 = func2.line(pts2)
    
    format_string = '{} function in {:d} dimensions: {:.3f} +/- {:.3f}'
    print(format_string.format('Gauss', ndims1, np.mean(value1), np.std(value1)/np.sqrt(npts)))
    print(format_string.format('Camel', ndims1, np.mean(value2), np.std(value2)/np.sqrt(npts)))
    print(format_string.format('Tsuda', ndims1, np.mean(value3), np.std(value3)/np.sqrt(npts)))
    print(format_string.format('Line', ndims1, np.mean(value4), np.std(value4)/np.sqrt(npts)))
    print(format_string.format('Gauss', ndims2, np.mean(value5), np.std(value5)/np.sqrt(npts)))
    print(format_string.format('Camel', ndims2, np.mean(value6), np.std(value6)/np.sqrt(npts)))
    print(format_string.format('Circle', ndims2, np.mean(value7), np.std(value7)/np.sqrt(npts)))
    print(format_string.format('Line', ndims2, np.mean(value8), np.std(value8)/np.sqrt(npts)))


if __name__ == '__main__':
    app.run(main)
