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
    
    
def run_vegas(func, ndims, ptspepoch, epochs):
    """ running VEGAS for integral + stddev vs. training
    
    Args:
    func:            integrand
    ndims (int):     dimensionality of the integrand
    ptspepoch (int): number of points per epoch in training
    epochs (int):    number of epochs for training
    ptsint (int):    number of points used to plot integral and uncertainty 
    """
    

    integ = vegas.Integrator(ndims*[[0.,1.]])

    x_values = []
    means = []
    stddevs = []

    mean, stddev = evaluate(func, integ, ptspepoch)
    print("Result = {:.3f} +/- {:.3f} ".format(mean, stddev))
    means.append(mean)
    stddevs.append(stddev)

    results = integ(func, nitn=epochs, neval=ptspepoch, adapt=True)#, max_nhcube = 1)
    print(results.summary())

    for number in results.itn_results:
        value, err = floatize(number)
        means.append(value)
        stddevs.append(err)
        #print("value = {} +/- {} ".format(value, err))

    #mean, stddev = evaluate(func, integ, ptsint)
    means = np.array(means)
    stddevs = np.array(stddevs)
    
    #print("Result = {:.3f} +/- {:.3f} ".format(mean, stddev))

    """
    for epoch in range(epochs):
        integ(func, nitn=1, neval=ptspepoch, adapt=True, max_nhcube = 1)
        mean, stddev = evaluate(func, integ, ptsint)
        x_values.append((epoch+1)*ptspepoch)
        means.append(mean)
        stddevs.append(stddev)
        print("Epoch {:d} of {:d} done".format(epoch+1, epochs))
    x_values = np.array(x_values)
    means = np.array(means)
    stddevs = np.array(stddevs)
    """
    return means, stddevs

def floatize(number):
    value = ""
    err = ""
    fullerr = ""
    switch = False
    for i in str(number):
        if not switch:
            value += i
            if i == ".":
                fullerr += "."
            else:
                fullerr += "0"
        else:
            err += i
        if i == "(":
            switch = True
            value = value[:-1]
        if i == ")":
            err = err[:-1]
    fullerr = fullerr[:-len(err)-1]+err
    return float(value), float(fullerr)

    
def evaluate(func, integ, ptsint=100000):
    """
    Takes an instance of Vegas and evaluates the integral
    without further adaptation

    Args:
    func:   Integrand to be integrated
    integ:  An instance of vegas.Integrator
    ptsint: Number of points used for evaluation
    """
    result = integ(func, nitn=1, neval=ptsint, adapt=False, max_nhcube = 1)
    return result.mean, result.sdev

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

    ndims = 6
    alpha = .1

    npts = 100000
    pts = np.random.rand(npts, ndims)

    # select function:
    func = TestFunctions(ndims, alpha)

    value = func.camel(pts)
    
    format_string = 'Crude MC of {} function in {:d} dimensions: {:.3f} +/- {:.3f}'
    print(format_string.format('Camel', ndims, np.mean(value), np.std(value)/np.sqrt(npts)))

    epochs = 50
    ptspepoch = 10000
    x_values = np.arange(0, (epochs + 1) * ptspepoch, ptspepoch)

    vegas_mean, vegas_err = run_vegas(func.camel, ndims, ptspepoch, epochs)
    plt.figure(dpi=150, figsize=[5., 4.])
    plt.xlim(0., epochs * ptspepoch)
    plt.xlabel('Evaluations in training')
    plt.ylim(0., 2.)
    plt.ylabel('Integral value')
    #plt.yscale('log')
    plt.plot(x_values, vegas_mean, color = 'b')
    plt.fill_between(x_values, vegas_mean + vegas_err, vegas_mean - vegas_err, color='b',alpha=0.5)
    plt.plot([0., epochs * ptspepoch], [1., 1.], ls = '--', color = 'k')
    plt.title('VEGAS integral')
    plt.show()

    plt.figure(dpi=150, figsize=[5., 4.])
    plt.xlim(0., epochs * ptspepoch)
    plt.xlabel('Evaluations in training')
    plt.ylim(1e-4, 1e1)
    plt.ylabel('Integral uncertainty')
    plt.yscale('log')
    plt.plot(x_values, vegas_err, color = 'b')
    plt.title('VEGAS uncertainty')
    plt.show()
    

if __name__ == '__main__':
    app.run(main)
