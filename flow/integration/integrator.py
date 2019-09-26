""" Implement the flow integrator. """
# pylint: disable=locally-disabled, invalid-name

import tensorflow as tf
import tensorflow_probability as tfp

from . import couplings

tfb = tfp.bijectors
tfd = tfp.distributions


def ewma(data, window):
    """
    Function to caluclate the Exponentially weighted moving average.

    Args:
        data (np.ndarray, float64): An array of data for the average to be
                                    calculated with.
        window (int64): The decay window.

    Returns:
        int64: The EWMA for the last point in the data array
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

    Metrics:
    -------
        - KL-Divergence
        - Chi2-Divergence

    """

    def __init__(self, func, dist, optimizer, **kwargs):
        """ Initialize the normalizing flow integrator. """
        self._func = func
        self.global_step = 0
        self.dist = dist
        self.optimizer = optimizer
        if 'loss_func' in kwargs:
            if kwargs['loss_func'] == 'chi2':
                self.loss_func = lambda true, test, logq, logp: tf.reduce_mean(
                    input_tensor=(true - test)**2/test**2)
                self.grad = lambda true, test, logq, logp: tf.reduce_mean(
                    input_tensor=-tf.stop_gradient(
                        (true/test)**2)*logq)
            elif kwargs['loss_func'] == 'kl':
                self.loss_func = lambda true, test, logq, logp: tf.reduce_mean(
                    input_tensor=(true/test)*(logp-logq))
                self.grad = lambda true, test, logq, logp: tf.reduce_mean(
                    input_tensor=-tf.stop_gradient(
                        (true/test))*logq)
            else:
                raise NotImplementedError('Requested loss_func: {}, '
                                          'is not implemented'.format(
                                              kwargs['loss_func']))
        else:
            self.loss_func = lambda true, test, logq, logp: tf.reduce_mean(
                input_tensor=(true - test)**2/test**2)
            self.grad = lambda true, test, logq, logp: tf.reduce_mean(
                input_tensor=-tf.stop_gradient(
                    (true/test)**2)*logq)

    @tf.function
    def train_one_step(self, nsamples, integral=False):
        """ Preform one step of integration and improve the sampling. """
        with tf.GradientTape() as tape:
            samples = tf.stop_gradient(self.dist.sample(nsamples))
            logq = self.dist.log_prob(samples)
            test = self.dist.prob(samples)
            true = self._func(samples)
            mean, var = tf.nn.moments(x=true/test, axes=[0])
            true = true/mean
            logp = tf.where(true > 1e-16, tf.math.log(true),
                            tf.math.log(true+1e-16))
            loss = self.loss_func(true, test, logq, logp)
            grad = self.grad(true, test, logq, logp)

        grads = tape.gradient(grad, self.dist.trainable_variables)
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

#    @tf.function
    def acceptance_calc(self, accuracy):
        """ Calculate the acceptance using a right tailed confidence interval
        with an accuracy of accuracy. """

        eta = 1
        eta_0 = 0
        weights = []
        tf_weights = tf.zeros(0)
        while abs(eta - eta_0)/eta > 0.01:
            nsamples = (tf.cast(tf.math.ceil(accuracy**-2/eta),
                                dtype=tf.int32)
                        - len(tf_weights))
            print(nsamples)
            iterations = tf.ones(nsamples // 10000, dtype=tf.int32)*10000
            remainder = tf.cast(tf.range(nsamples // 10000) < nsamples % 10000,
                                tf.int32)
            print(iterations, remainder)
            iterations += remainder
            for iteration in iterations:
                print(iteration)
                weights.append(self.acceptance(iteration.numpy()))
            tf_weights = tf.sort(tf.concat(weights, axis=0))
            cum_weights = tf.cumsum(tf_weights)
            cum_weights /= cum_weights[-1]
            index = tf.cast(
                tf.searchsorted(cum_weights,
                                tf.convert_to_tensor([1-accuracy],
                                                     dtype=cum_weights.dtype)),
                dtype=tf.int32)
            max_val = tf_weights[index[0]]
            avg_val = tf.reduce_mean(tf_weights[:index[0]])
            eta_0 = eta
            eta = avg_val/max_val
            print(eta_0, eta)

        return avg_val/max_val

    def save(self):
        """ Save the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.save_weights(
                './models/model_layer_{:02d}'.format(j))

    def load(self):
        """ Load the network. """
        for j, bijector in enumerate(self.dist.bijector.bijectors):
            bijector.transform_net.load_weights(
                './models/model_layer_{:02d}'.format(j))
        print("Model loaded successfully")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import corner
    import numpy as np

    class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
        """ Implement CosineAnnealing. """
        def __init__(self, base_lr, total_epochs, eta_min=0):
            self.base_lr = base_lr
            self.total_epochs = total_epochs
            self.eta_min = eta_min

        def __call__(self, step):
            frac_epochs = step / self.total_epochs
            return self.eta_min + (self.base_lr - self.eta_min) \
                * (1 + tf.math.cos(np.pi * frac_epochs)) / 2

        def call(self, step):
            """ Call the function. """
            return self(step)

    def build_dense(in_features, out_features):
        """ Build a dense NN. """
        invals = tf.keras.layers.Input(in_features)
        hidden = tf.keras.layers.Dense(128)(invals)
        hidden = tf.keras.layers.Dense(128)(hidden)
        hidden = tf.keras.layers.Dense(128)(hidden)
        hidden = tf.keras.layers.Dense(128)(hidden)
        hidden = tf.keras.layers.Dense(128)(hidden)
        hidden = tf.keras.layers.Dense(128)(hidden)
        outputs = tf.keras.layers.Dense(out_features)(hidden)
        model = tf.keras.models.Model(invals, outputs)
        model.summary()
        return model

    def func_(x):
        """ Test function 1. """
        return tf.reduce_prod(x, axis=-1)

    def camel(x):
        """ Camel test function. """
        return (tf.exp(-1./0.004*((x[:, 0]-0.25)**2+(x[:, 1]-0.25)**2))
                + tf.exp(-1./0.004*((x[:, 0]-0.75)**2+(x[:, 1]-0.75)**2)))

    def circle(x):
        """ Circle test function. """
        dx1 = 0.4
        dy1 = 0.6
        rr = 0.25
        w1 = 1./0.004
        ee = 3.0
        return (x[:, 1]**ee*tf.exp(-w1*tf.abs((x[:, 1]-dy1)**2
                                              + (x[:, 0]-dx1)**2-rr**2))
                + (1-x[:, 1])**ee
                * tf.exp(-w1*tf.abs((x[:, 1]-1.0+dy1)**2
                                    + (x[:, 0]-1.0+dx1)**2-rr**2)))

    def func2(x):
        """ Test function 2. """
        return tf.where((x[:, 0] < 0.9) & (x[:, 1] < 0.9),
                        (x[:, 0]**2 + x[:, 1]**2)/((1-x[:, 0])*(1-x[:, 1])), 0)

    def main():
        """ Main function. """
        ndims = 2
        epochs = int(500)

        bijectors = []
        bijectors.append(couplings.PiecewiseRationalQuadratic(
            [1, 0], build_dense, num_bins=100, blob=True))
        bijectors.append(couplings.PiecewiseRationalQuadratic(
            [0, 1], build_dense, num_bins=100, blob=True))

        bijectors = tfb.Chain(list(reversed(bijectors)))

        base_dist = tfd.Uniform(low=ndims*[0.], high=ndims*[1.])
        base_dist = tfd.Independent(distribution=base_dist,
                                    reinterpreted_batch_ndims=1,
                                    )
        dist_ = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=bijectors,
        )

        initial_learning_rate = 1e-4
        lr_schedule = CosineAnnealing(initial_learning_rate, epochs)

        optimizer_ = tf.keras.optimizers.Adam(
            lr_schedule, clipnorm=5.0)  # lr_schedule)

        integrator = Integrator(func2, dist_, optimizer_)
        losses = []
        integrals = []
        errors = []
        min_loss = 1e99
        nsamples_ = 5000
        try:
            for epoch in range(epochs):
                if epoch % 5 == 0:
                    samples_ = integrator.sample(10000)
                    hist2d_kwargs = {'smooth': 2}
                    figure = corner.corner(samples_,
                                           labels=[r'$x_1$', r'$x_2$'],
                                           show_titles=True,
                                           title_kwargs={"fontsize": 12},
                                           range=ndims*[[0, 1]],
                                           **hist2d_kwargs)

                loss_, integral_, error = integrator.train_one_step(
                    nsamples_, True)
                if epoch % 5 == 0:
                    figure.suptitle('loss = '+str(loss_.numpy()),
                                    fontsize=16, x=0.75)
                    plt.savefig('fig_{:04d}.png'.format(epoch))
                    plt.close()
                losses.append(loss_)
                integrals.append(integral_)
                errors.append(error)
                if loss_ < min_loss:
                    min_loss = loss_
                    integrator.save()
                if epoch % 10 == 0:
                    print(epoch, loss_.numpy(), integral_.numpy(),
                          error.numpy())
        except KeyboardInterrupt:
            pass

        integrator.load()

        weights = []
        for _ in range(10):
            weights.append(integrator.acceptance(100000).numpy())
        weights = np.concatenate(weights)

        # Remove outliers
        weights = np.sort(weights)
        weights = np.where(weights < np.mean(weights)*0.01, 0, weights)

        average = np.mean(weights)
        max_wgt = np.max(weights)

        print("acceptance = "+str(average/max_wgt))

        plt.hist(weights, bins=np.logspace(-2, 2, 100))
        plt.axvline(average, linestyle='--', color='red')
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('efficiency.png')
        plt.show()

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('loss.png')
        plt.show()

    main()
