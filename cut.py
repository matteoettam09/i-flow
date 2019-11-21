""" Test cut efficiency """

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from descartes.patch import PolygonPatch
import corner

from flow.integration import integrator
from flow.integration import couplings
from flow.splines.spline import _knot_positions, _search_sorted
from flow.splines.spline import _gather_squeeze

tfd = tfp.distributions  # pylint: disable=invalid-name
tf.keras.backend.set_floatx('float64')

CUT_VALUE = 0.05
ALPHA = 1.0
COLOR = ['red', 'magenta', 'green', 'blue', 'black']


def func(pts_x):
    """ Calculate function for testing. """
    return tf.where(pts_x[:, 0] > CUT_VALUE, tf.pow(pts_x[:, 0], -ALPHA), 0)


class Cheese:
    """ Class to store the cheese function. """

    def __init__(self, nholes):
        """ Init cheese function holes. """

        # Create random holes
        self.position = np.random.random((nholes, 2))
        self.radius = 0.1*np.random.random(nholes)+0.05

        # Create shape
        holes = Point(self.position[0]).buffer(self.radius[0])
        for i in range(1, nholes):
            circle = Point(self.position[i]).buffer(self.radius[i])
            holes = holes.union(circle)

        self.cheese = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        self.cheese = self.cheese.symmetric_difference(holes)

    def __call__(self, pts):
        """ Calculate a swiss cheese like function. """
        mask = np.zeros_like(pts[:, 0], dtype=np.float64)
        for i, position in enumerate(pts):
            point = Point(position[0], position[1])
            mask[i] = float(self.cheese.contains(point))

        return mask

    def plot(self, pts=None, filename=None):
        """ Plot the cheese. """
        patch = PolygonPatch(self.cheese, facecolor='yellow',
                             alpha=0.5, zorder=1)
        fig = plt.figure()
        axis = fig.add_subplot(111)
        if pts is not None:
            plt.scatter(pts[:, 0], pts[:, 1], s=5, zorder=2)
        axis.add_patch(patch)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        if filename is not None:
            plt.savefig('{}.png'.format(filename))
        plt.show()

    @property
    def area(self):
        """ Get the area of cheese surface. """
        return self.cheese.area


class Ring:
    """ Class to store the ring function. """

    def __init__(self, radius1, radius2):
        """ Init ring function. """

        # Ensure raidus1 is the large one
        if radius1 < radius2:
            radius1, radius2 = radius2, radius1

        # Create shape
        self.ring = Point((0.5, 0.5)).buffer(radius1)
        hole = Point((0.5, 0.5)).buffer(radius2)
        self.ring = self.ring.symmetric_difference(hole)

    def __call__(self, pts):
        """ Calculate a swiss ring like function. """
        mask = np.zeros_like(pts[:, 0], dtype=np.float64)
        for i, position in enumerate(pts):
            point = Point(position[0], position[1])
            mask[i] = float(self.ring.contains(point))

        return mask

    def plot(self, pts=None, filename=None, lines=None):
        """ Plot the ring. """
        patch = PolygonPatch(self.ring, facecolor='red',
                             alpha=0.5, zorder=1)
        fig = plt.figure()
        axis = fig.add_subplot(111)
        if pts is not None:
            plt.scatter(pts[:, 0], pts[:, 1], s=5, zorder=2)
        if lines is not None:
            for i in range(5):
                position = float(i)/10.0 + 0.1
                plt.axvline(x=position, color=COLOR[i])
        axis.add_patch(patch)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        if filename is not None:
            plt.savefig('{}.png'.format(filename))
        plt.show()

    @property
    def area(self):
        """ Get the area of ring surface. """
        return self.ring.area


def get_spline(inputs, widths, heights, derivatives):
    """ Get the points of the splines to plot. """
    min_bin_width = 1e-15
    min_bin_height = 1e-15
    min_derivative = 1e-15

    num_bins = widths.shape[-1]

    widths = tf.nn.softmax(widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = _knot_positions(widths, 0)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = ((min_derivative + tf.nn.softplus(derivatives))
                   / (tf.cast(min_derivative + tf.math.log(2.), tf.float64)))

    heights = tf.nn.softmax(heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = _knot_positions(heights, 0)
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = _search_sorted(cumwidths, inputs)

    input_cumwidths = _gather_squeeze(cumwidths, bin_idx)
    input_bin_widths = _gather_squeeze(widths, bin_idx)

    input_cumheights = _gather_squeeze(cumheights, bin_idx)
    delta = heights / widths
    input_delta = _gather_squeeze(delta, bin_idx)

    input_derivatives = _gather_squeeze(derivatives, bin_idx)
    input_derivatives_p1 = _gather_squeeze(derivatives[..., 1:], bin_idx)

    input_heights = _gather_squeeze(heights, bin_idx)

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (input_delta * theta**2
                                 + input_derivatives
                                 * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_p1
                                  - 2 * input_delta)
                                 * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator

    return outputs, cumwidths, cumheights


def plot_spline(widths, heights, derivatives, color):
    """ Plot the spline. """
    nsamples = 10000
    # nodes = 5

    pts_x = np.linspace(0, 1, nsamples).reshape(nsamples, 1)
    widths = np.array([widths.numpy().tolist()]*nsamples)
    heights = np.array([heights.numpy().tolist()]*nsamples)
    derivatives = np.array([derivatives.numpy().tolist()]*nsamples)

    outputs, widths, heights = get_spline(pts_x, widths, heights, derivatives)

    plt.plot(pts_x, outputs, zorder=1, color=color)
    plt.scatter(widths.numpy(), heights.numpy(), s=20, color='red', zorder=2)
    # plt.axhline(y=CUT_VALUE)
    # plt.axvline(x=CUT_VALUE)


def build(in_features, out_features, options):
    " Build the NN. """
    del options

    invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
    hidden = tf.keras.layers.Dense(128, activation='relu')(invals)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    hidden = tf.keras.layers.Dense(128, activation='relu')(hidden)
    outputs = tf.keras.layers.Dense(out_features, bias_initializer='zeros',
                                    kernel_initializer='zeros')(hidden)
    model = tf.keras.models.Model(invals, outputs)
    model.summary()
    return model


def one_blob(xd, nbins_in):
    """ Perform one_blob encoding. """
    num_identity_features = xd.shape[-1]
    y = tf.tile(((0.5/nbins_in) + tf.range(0., 1.,
                                           delta=1./nbins_in)),
                [tf.size(xd)])
    y = tf.cast(tf.reshape(y, (-1, num_identity_features,
                               nbins_in)),
                dtype=tf.float64)
    res = tf.exp(((-nbins_in*nbins_in)/2.)
                 * (y-xd[..., tf.newaxis])**2)
    res = tf.reshape(res, (-1, num_identity_features*nbins_in))
    return res


def main():
    """ Main function """
    quadratic = False
    tf.config.experimental_run_functions_eagerly(True)
    cheese = Ring(0.5, 0.2)
    print("Actual area is {}".format(cheese.area))
    bijectors = []
    """
    bijector = couplings.PiecewiseRationalQuadratic([1, 0], build,
                                                    num_bins=10,
                                                    blob=None,
                                                    options=None)
    """
    if quadratic:
        bijectors.append(couplings.PiecewiseQuadratic([1, 0], build,
                                                    num_bins=10,
                                                    blob=None,
                                                    options=None))
        bijectors.append(couplings.PiecewiseQuadratic([0, 1], build,
                                                    num_bins=10,
                                                    blob=None,
                                                    options=None))
    else:
        bijectors.append(couplings.PiecewiseRationalQuadratic([1, 0], build,
                                                    num_bins=10,
                                                    blob=None,
                                                    options=None))
        bijectors.append(couplings.PiecewiseRationalQuadratic([0, 1], build,
                                                    num_bins=10,
                                                    blob=None,
                                                    options=None))

    bijector = tfp.bijectors.Chain(list(reversed(bijectors)))
    low = np.array([0, 0], dtype=np.float64)
    high = np.array([1, 1], dtype=np.float64)
    dist = tfd.Uniform(low=low, high=high)
    dist = tfd.Independent(distribution=dist,
                           reinterpreted_batch_ndims=1)
    dist = tfd.TransformedDistribution(
        distribution=dist,
        bijector=bijector)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        2e-3, decay_steps=75, decay_rate=0.5)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=10.0)
    integrate = integrator.Integrator(cheese, dist, optimizer,
                                      loss_func='exponential')
    if not quadratic:
        num = 0
        for elem in dist.bijector.bijectors:
        
            for i in range(5):
                point = float(i)/10.0 + 0.1
                # transform_params = bijector.transform_net(
                #     one_blob(np.array([[point]]), 16))
                #transform_params = bijector.transform_net(np.array([[point]]))
                transform_params = elem.transform_net(np.array([[point]]))
            
                widths = transform_params[..., :10]
                heights = transform_params[..., 10:20]
                derivatives = transform_params[..., 20:]
                plot_spline(widths, heights, derivatives, COLOR[i])

            plt.savefig('pretraining_{}.png'.format(num))
            plt.show()
            num += 1
    
    cheese.plot(filename='cheese', lines=True)

    for epoch in range(300):
        loss, integral, error = integrate.train_one_step(8000,
                                                         integral=True)
        if epoch % 1 == 0:
            print('Epoch: {:3d} Loss = {:8e} Integral = '
                  '{:8e} +/- {:8e}'.format(epoch, loss, integral, error))
    if not quadratic:
        num = 0    
        for elem in dist.bijector.bijectors:
            for i in range(5):
                point = float(i)/10.0 + 0.1
                # transform_params = bijector.transform_net(
                #     one_blob(np.array([[point]]), 16))
                #transform_params = bijector.transform_net(np.array([[point]]))
                transform_params = elem.transform_net(np.array([[point]]))
                widths = transform_params[..., :10]
                heights = transform_params[..., 10:20]
                derivatives = transform_params[..., 20:]
                plot_spline(widths, heights, derivatives, COLOR[i])

            plt.savefig('posttraining_{}.png'.format(num))
            num += 1
            plt.show()
    
    nsamples = 50000
    hist2d_kwargs = {'smooth': 2, 'plot_datapoints': False}
    pts = integrate.sample(nsamples)
    figure = corner.corner(pts, labels=[r'$x_{{{}}}$'.format(x)
                                        for x in range(2)],
                           show_titles=True,
                           title_kwargs={'fontsize': 12},
                           range=2*[[0, 1]],
                           **hist2d_kwargs)
    plt.savefig('ring_corner.png')
    plt.show()
    plt.close()

    # todo:
    # second dim
    # corner visualization
    # check quadratic

if __name__ == '__main__':
    main()
