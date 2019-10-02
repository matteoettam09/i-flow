""" An example on how to use the flow integrator. """

# pylint: disable=invalid-name

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import corner
from flow.integration import couplings
from flow.integration import integrator

tfb = tfp.bijectors
tfd = tfp.distributions


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

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements


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

    integral = integrator.Integrator(func2, dist_, optimizer_)
    losses = []
    integrals = []
    errors = []
    min_loss = 1e99
    nsamples_ = 5000
    try:
        for epoch in range(epochs):
            if epoch % 5 == 0:
                samples_ = integral.sample(10000)
                hist2d_kwargs = {'smooth': 2}
                figure = corner.corner(samples_,
                                       labels=[r'$x_1$', r'$x_2$'],
                                       show_titles=True,
                                       title_kwargs={"fontsize": 12},
                                       range=ndims*[[0, 1]],
                                       **hist2d_kwargs)

            loss_, integral_, error = integral.train_one_step(
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
                integral.save()
            if epoch % 10 == 0:
                print(epoch, loss_.numpy(), integral_.numpy(),
                      error.numpy())
    except KeyboardInterrupt:
        pass

    integral.load()

    weights = []
    for _ in range(10):
        weights.append(integral.acceptance(100000).numpy())
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


if __name__ == '__main__':
    main()
