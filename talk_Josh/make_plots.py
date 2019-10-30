""" Generate figures for LPC talk. """
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma


def integration_plots():
    """Generate plots for integration demonstration."""
    pts = np.random.random((2, 200))
    inside = pts[0]**2 + pts[1]**2 < 1
    inside = pts[0]**2 + pts[1]**2 < 1
    outside = pts[0]**2 + pts[1]**2 > 1
    pts_in = pts[:, inside]
    pts_out = pts[:, outside]

    circle = plt.Circle((0, 0), 1, ls='-', lw=2, edgecolor='black',
                        facecolor='white', fill=False)

    _, axis = plt.subplots()
    axis.add_artist(circle)
    axis.scatter(pts_in[0], pts_in[1], color='red')
    axis.scatter(pts_out[0], pts_out[1], color='blue')

    plt.savefig('figs/circle.pdf', bbox_inches='tight')
    plt.clf()

    x_pts = np.linspace(0, 1, 101)
    y_pts = x_pts**2

    x_bins = np.linspace(-0.1, 1, 12)
    y_bins = x_bins**2

    plt.plot(x_pts, y_pts)
    plt.plot(x_bins[1:], y_bins[:-1], drawstyle='steps', color='red')
    plt.xlim([0, 1])

    plt.savefig('figs/midpoint.pdf', bbox_inches='tight')
    plt.clf()

    x_pts = np.linspace(0, 1, 100)
    y_pts = np.ones_like(x_pts)
    plt.plot(x_pts, y_pts, lw=3)
    plt.tick_params(axis='both', reset=True, which='both', direction='in')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.savefig('figs/easy.pdf', bbox_inches='tight')
    plt.clf()

    y_pts = (-x_pts**3*np.exp(-(x_pts**2/(2*100)))
             + x_pts**2*np.exp(-(x_pts**2)/(2*50)))*7
    plt.plot(x_pts, y_pts, lw=3)
    plt.tick_params(axis='both', reset=True, which='both', direction='in')
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.savefig('figs/hard.pdf', bbox_inches='tight')
    plt.clf()


def dimensionality_plots():
    """ Generate dimensionality plots. """
    dim = np.arange(1, 20)
    ratio = np.pi**(dim/2.)/(dim*2**(dim-1)*gamma(dim/2))

    plt.plot(dim, ratio, lw=2)
    plt.tick_params(axis='both', reset=True, which='both', direction='in')
    plt.xlabel('Dimensionality', fontsize=16)
    plt.ylabel(r'$\frac{V_{hypersphere}}{V_{hypercube}}$', fontsize=18)
    plt.savefig('figs/dimensionality.pdf', bbox_inches='tight')
    plt.clf()


def jacobian_plots():
    """ Generate Jacobian plots. """
    x_pts = np.linspace(0, 1, 100)
    y_pts = x_pts**2
    plt.plot(x_pts, y_pts, lw=3)
    plt.tick_params(axis='both', reset=True, which='both', direction='in')
    plt.savefig('figs/func.pdf', bbox_inches='tight')
    plt.clf()

    y_pts = 1./(2*x_pts)
    plt.plot(x_pts, y_pts, lw=3)
    plt.tick_params(axis='both', reset=True, which='both', direction='in')
    plt.savefig('figs/jac.pdf', bbox_inches='tight')
    plt.clf()


def main():
    """ Main driving code. """
    integration_plots()
    dimensionality_plots()
    jacobian_plots()


if __name__ == '__main__':
    main()
