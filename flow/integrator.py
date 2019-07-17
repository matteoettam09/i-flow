# -*- coding: utf-8 -*-
"""Implement the Flow Integrator.

This module implements a driver for performing integration on an aribitrary function.
The module contains a bijector factory for allowing flexibility in the layers the user
has for the integration. 

Exmaple:
   Exampels can be found in the examples folder. To running all of them can be done by:
        $ python examples_integrate.py

Attributes:
    factor (BijectorFactory): Module level factory variable used to store all the 
    different coupling layers in a factory class. This allows the user to easily
    select the desired configuration of the integrator.

Todo:
    * Clean up options
    * Allow more flexibility in plotting
    * Improve the speed of running
    * Figure out how to appropriately permute dimensions for high dimensionality problems
    * Save the best network and reload at the end
    * Allow for MPI training
"""

import os
import corner
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import flow.piecewise as piecewise
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


def ewma(data, window):
    """
    Function to caluclate the Exponentially weighted moving average.

    Args:
        data (np.ndarray, float64): An array of data for the average to be calculated with.
        window (int64): The decay window.

    Returns:
        int64: The EWMA for the last point in the data array
    """
    if len(data) < window:
        return data[-1]

    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(data, weights, mode='full')[:len(data)]
    a[:window] = a[window]
    return a[-1]


class BijectorFactory:

    """Implement a bijector factory.

    The integrator requires bijector layers for training the integrator. To
    allow the user flexibility in the integrator, the different bijector layers 
    are loaded into a factory class. This allows for easier creation of the 
    integrator object.

    """

    def __init__(self):
        """Create an empty dictionary for the different bijectors."""
        self._bijectors = {}

    def register_bijector(self, key, bijector):
        """Register a new bijector layer to the factory.
        
        Args:
            key (str): Name of the bijector layer (Ex: 'linear')
            bijector (tfb.bijector): A bijector class object
        """
        self._bijectors[key] = bijector

    def create(self, key, **kwargs):
        """Create a new bijector object given a key.

        Args:
            key (str): Type of bijector layer to be created
            **kwargs: A list of options to be passed to the bijector constructor

        Returns:
            tfb.bijector: A newly created bijector layer

        Raises:
            ValueError: If the requested bijector is not valid
        """
        bijector = self._bijectors.get(key)
        if not bijector:
            raise ValueError(key)
        return bijector(**kwargs)


# Initialize the factory class and fill with allowed layers
factory = BijectorFactory()
factory.register_bijector('linear', piecewise.PiecewiseLinear)
factory.register_bijector('quadratic', piecewise.PiecewiseQuadratic)
factory.register_bijector('quadratic_const', piecewise.PiecewiseQuadraticConst)


class Integrator():

    """Integrate a given function using normalizing flow neural networks.

    The network is setup to integrate a user defined function given with a fixed
    number of input dimensions in the unit hypercube. 

    """

    def __init__(self, func, ndims, layers=4,
                 mode='quadratic', name = None, **kwargs):
        """Initialize the integrator class.

        Setup all the needed options for the integrator, such as: the function to evaluate,
        the number of dimensions, the type of layers, the number of layers, and the number of
        bins in each layer. The initialization then creates the model to be used for the
        integrator. The model is then used to create a transformed distribution from a 
        uniform distribution into the learned distribution.

        Args:
            func (function): The function to be integrated.
            ndims (int): Number of dimensions to integrate over.
            layers (int): The number of layers for the network to have (not used currently).
            mode (str): String to determine which bijector to use.
            name (str): Name of the integral, used for saving/loading the model and other files.
            kwargs: Additional arguments to be passed to the bijector layer.
        """
        self.func = func
        self.ndims = ndims
        self.mode = mode
        self.layers = layers

        self.losses = []
        self.integrals = []
        self.vars = []
        self.global_step = 0

        self.bijectors = []

        self.labels = [r'$x_{}$'.format(i) for i in range(self.ndims)]

        arange = np.arange(ndims)
        permute = np.hstack([arange[1:], arange[0]])
        kwargs['D'] = ndims
        kwargs['d'] = ndims//2
        if 'nbins' not in kwargs:
            kwargs['nbins'] = 25
        for i in range(ndims):
            kwargs['layer_id'] = i
            self.bijectors.append(factory.create(mode, **kwargs))
            self.bijectors.append(tfb.Permute(permutation=permute))

        self.bijectors = tfb.Chain(list(reversed(self.bijectors)))

        self.base_dist = tfd.Uniform(low=ndims*[0.], high=ndims*[1.])
        self.base_dist = tfd.Independent(distribution=self.base_dist,
                                         reinterpreted_batch_ndims=1,
                                         )

        self.dist = tfd.TransformedDistribution(
            distribution=self.base_dist,
            bijector=self.bijectors,
        )

        self.name = name
        self.saver = tf.train.Saver()

    def _loss_fn(self, nsamples, alpha):
        x = self.dist.sample(nsamples)
        logq = self.dist.log_prob(x)
        p = self.func(x)
        q = self.dist.prob(x)
        xsec = p/q
        p = p/tf.reduce_mean(xsec)
        mean, var = tf.nn.moments(xsec, axes=[0])
        acceptance = tf.reduce_mean(xsec)/tf.reduce_max(xsec)
#        return (1.0/acceptance**2, mean,
#                var/nsamples, x, p, q)
#        return (tf.reduce_mean(p/q*(tf.log(p)-logq))/acceptance**2, mean,
#                var/nsamples, x, p, q)
        return ((1-alpha)*tf.reduce_mean(p/q*(tf.log(p)-logq))-alpha*acceptance, mean,
                var/nsamples, x, p, q)
#        return (tf.reduce_mean(p/q*(tf.log(p)-logq)), mean,
#                var/nsamples, x, p, q)

    def make_optimizer(self, learning_rate=1e-4, nsamples=500, alpha=0):
        """Create the optimizer.

        Args:
            learning_rate (float): Initial learning rate for the optimizer.
            nsamples (int): Number of samples to use when estimating the loss.
        """
        (self.loss, self.integral, self.var,
            self.x, self.p, self.q) = self._loss_fn(nsamples,alpha)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        self.opt_op = optimizer.apply_gradients(grads)

    def optimize(self, sess, **kwargs):
        """Optimize the integrator for the give function.

        Preform the training of the neural network for a given number of epochs.
        The training runs for a given number of epochs using the gradient to minimize
        the loss function (KL-divergence).

        Note: To end the training early, but continue running the code, use the command
        <ctrl>-c.

        Args:
            sess (tf.Session): Tensorflow session that contains the neural network.
            **kwargs: Additional options to set for optimizing. Options are:
                epochs (int): Number of epochs to train for (default is 1000).
                printout (int): How often to print results (default is 100).
                profiler (tf.profiler): Tensorflow profiler for profiling the code.
                options (tf.options): Tensorflow options for the profiler.
        """
        # Break out the possible keyword arguments
        epochs = kwargs.get('epochs',1000)
        printout = kwargs.get('printout',100)
        profiler = kwargs.get('profiler')
        if profiler is not None:
            options = kwargs.get('options')
        else:
            options = None
        plot = kwargs.get('plot',False)

        min_loss = 1e99
        if self.name is not None:
            try:
                self.load(sess,'models/'.format(self.name))
            except:
                pass

        # Preform training
        try:
            for epoch in range(epochs):
                if profiler is not None:
                    run_metadata = tf.RunMetadata()
                else:
                    run_metadata = None
                (_, np_loss, np_integral,
                 np_var, xpts, ppts, qpts) = sess.run([self.opt_op, self.loss,
                                                       self.integral, self.var,
                                                       self.x, self.p, self.q],
                                                      options=options,
                                                      run_metadata=run_metadata)
                if profiler is not None:
                    profiler.add_step(epoch, run_metadata)
                self.global_step += 1
                if np_loss < min_loss and self.name is not None:
                    self.save(sess,'models/')
                    min_loss = np_loss
                self.losses.append(np_loss)
                self.integrals.append(np_integral)
                self.vars.append(np_var)
                if epoch % printout == 0:
                    print("Epoch {:4d}: loss = {:e}, integral = {:e} +/- {:e}".format(
                        epoch, np_loss, ewma(self.integrals, 10),
                        np.sqrt(ewma(self.vars, 10)))
                    )
                    if plot:
                        figure = corner.corner(xpts, labels=self.labels,
                                               show_titles=True, title_kwargs={"fontsize": 12},
                                               range=self.ndims*[[0, 1]]
                                               )
                        plt.savefig('fig_{:04d}.pdf'.format(epoch))
                        plt.close()
        except KeyboardInterrupt:
            print('Caught Crtl-C. Ending training early.')

        print("Epoch {:4d}: loss = {:e}, integral = {:e} +/- {:e}".format(
            epoch, np_loss, ewma(self.integrals, 10),
            np.sqrt(ewma(self.vars, 10)))
        )
        if plot:
            figure = corner.corner(xpts, labels=self.labels, show_titles=True,
                                   title_kwargs={"fontsize": 12}, range=self.ndims*[[0, 1]]
                                   )
            plt.savefig('fig_{:04d}.pdf'.format(epoch))
            plt.close()

    def save(self, sess, path = 'models/'):
        """Save the model file."""
        save_path = self.saver.save(sess, os.path.join(path,'{}.ckpt'.format(self.name)))
        print("Model saved at: {}".format(save_path))

    def load(self, sess, path = 'models/'):
        """Load the saved model file."""
        self.saver.restore(sess,  os.path.join(path,'{}.ckpt'.format(self.name)))
        print("Model restored")

    def _plot(self, axis, labelsize=17, titlesize=20):
        axis.set_xlabel('epoch', fontsize=titlesize)
        axis.tick_params(axis='both', reset=True, which='both',
                         direction='in', size=labelsize)
        return axis

    def plot_loss(self, axis, labelsize=17, titlesize=20, start=0):
        """Plot the loss as a function of the epoch."""
        axis.plot(self.losses[start:])
        axis.set_ylabel('loss', fontsize=titlesize)
        axis.set_yscale('log')
        axis = self._plot(axis, labelsize, titlesize)
        return axis

    def plot_integral(self, axis, labelsize=17, titlesize=20, start=0):
        """Plot the integral as a function of the epoch."""
        axis.plot(self.integrals[start:])
        axis.set_ylabel('integral', fontsize=titlesize)
        axis = self._plot(axis, labelsize, titlesize)
        return axis

    def plot_variance(self, axis, labelsize=17, titlesize=20, start=0):
        """Plot the variance as a function of the epoch."""
        axis.plot(self.vars[start:])
        axis.set_ylabel('variance', fontsize=titlesize)
        axis.set_yscale('log')
        axis = self._plot(axis, labelsize, titlesize)
        return axis

    def integrate(self, sess, nsamples=10000, plot=False, acceptance=False, **kwargs):
        """Integrate the given function using the network.

        Preforms the integration of the function using the network. Ideally, this is
        called after the network has been trained. However, it is not required to be
        called after training. Calling it before training can allow the user to see 
        the improvement from the training more easily.

        Args:
            sess (tf.Session): The tensorflow session to be used for evaluating the network.
            nsamples (int): The number of points for evaluating the integral.
            plot (bool): Flag to plot 2-D and 1-D projections of the variables.
            acceptance (bool): Flag to plot and calculate the acceptance for unweighting.
            **kwargs: Additional options to be used for the code. Options are:
                min (float): minimum value for the acceptance plot.
                max (float): maximum value for the acceptance plot.
                nbins (int): number of bins to be used in the acceptance plot.
                plot_kwargs: Additional kwargs to be passed to the matplotlib plotting routine.

        Returns:
            integral (float): Estimated value for the integral
            error (float): Error estimate of the integral
            acceptance (float): Optional return of the average acceptance rate

        """
        x = self.dist.sample(nsamples)
        q = self.dist.prob(x)
        p = self.func(x)
        integral, var = tf.nn.moments(p/q, axes=[0])
        error = tf.sqrt(var/nsamples)

        tf_results = [integral, error]
        if plot:
            tf_results.append(x)
        if acceptance:
            r = p/(q*integral)
            tf_results.append(r)

        results = sess.run(tf_results)
        if plot:
            figure = corner.corner(results[2], labels=self.labels,
                                   show_titles=True, title_kwargs={"fontsize": 12},
                                   range=self.ndims*[[0, 1]]
                                   )
            plt.savefig('xsec_final.pdf')
            plt.close()

        if acceptance:
            # Load options
            min_val = kwargs.get('min',1e-7)
            max_val = kwargs.get('max',10)
            nbins = kwargs.get('nbins',100)
            path = kwargs.get('path',os.getcwd())

            if 'fname' in kwargs:
                fname = '_{}'.format(kwargs['fname'])
            else:
                fname = ''

            np.savetxt(os.path.join(path,'weights{}.txt'.format(fname)),results[-1],delimiter='\n')

            plt.hist(results[-1], bins=np.logspace(np.log10(min_val),
                                                   np.log10(max_val),
                                                   nbins), **kwargs.get('plot_kwargs',{}))
            plt.yscale('log')
            plt.xscale('log')
            plt.savefig(os.path.join(path,'acceptance{}.pdf'.format(fname)))
            plt.close()
            return results[0], results[1], np.mean(results[-1])/np.max(results[-1])

        return results[0], results[1]


if __name__ == '__main__':
    import tensorflow as tf
    import multiprocessing

    def normalChristina(x):
        """Example function to integrate."""
        return 0.8 * tf.exp(
            (-0.5*((x[:, 0]-0.5) * (50 * (x[:, 0]-0.5)
                - 15 * (x[:, 1]-0.5)) + (-15*(x[:, 0]-0.5) + 5*(x[:, 1]-0.5))
                   * (x[:, 1]-0.5)))) + x[:, 2]

    acceptances = []
    errors = []
    alphas = np.logspace(-5,0,100)
    
    def alpha_scan(alpha,error_send,acceptance_send):
        name = 'test_func_{}'.format(alpha)
        integrator = Integrator(
            normalChristina, 3, mode='linear', unet=True, blob=True, name=name)

        integrator.make_optimizer(nsamples=5000,alpha=alpha)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                integrator.load(sess)
            except:
                integrator.optimize(sess, epochs=2000)
                integrator.load(sess)
            integral, error, acceptance = integrator.integrate(sess, 100000, acceptance=True)
            error_send.send(error)
            acceptance_send.send(acceptance)

    for alpha in alphas:
        error_recv, error_send = multiprocessing.Pipe(False)
        acceptance_recv, acceptance_send = multiprocessing.Pipe(False)
        p = multiprocessing.Process(target=alpha_scan, args=(alpha,error_send,acceptance_send,))
        errors.append(error_recv)
        acceptances.append(acceptance_recv)
        p.start()
        p.join()

    errors = [x.recv() for x in errors]
    acceptances = [x.recv() for x in acceptances]

    print(acceptances)
    print(errors)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(16,5))
    ax1.plot(alphas,acceptances)
    ax2.plot(alphas,errors)
    ax3.plot(acceptances,errors)

    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'Acceptance')
    ax1.set_xscale('log')

    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'Error')
    ax2.set_xscale('log')

    ax2.set_xlabel(r'Acceptance')
    ax2.set_ylabel(r'Error')

    plt.show()

    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    #ax1 = integrator.plot_loss(ax1)
    #ax2 = integrator.plot_integral(ax2)
    #ax3 = integrator.plot_variance(ax3)
    #plt.show()
