""" Implement the Sinkhorn loss.

    Modified from the pyTorch implementation found here,
    https://github.com/gpeyre/SinkhornAutoDiff/tree/generalization
    into Tensorflow.
"""

import tensorflow as tf


@tf.function
def sinkhorn_normalized(true, test, eps, mu, nu, niter):
    """ Computes Sinkhorn divergence. """
    w_xy = sinkhorn_loss(true, test, eps, mu, nu, niter)
    w_xx = sinkhorn_loss(true, true, eps, mu, mu, niter)
    w_yy = sinkhorn_loss(test, test, eps, nu, nu, niter)

    return w_xy - 0.5 * w_xx - 0.5 * w_yy


@tf.function
def sinkhorn_loss(true, test, eps, mu, nu, niter=100,
                  acc=1e-3, unbalanced=False):
    """ Calculate the Sinkhorn loss. """
    C = cost_matrix(true, test)

    tau = -0.8
    thresh = acc

    if unbalanced:
        rho = (0.5)**2
        lam = rho / (rho + eps)

    def ave(u, u1):
        """ Barycenter subroutine. """
        return tau * u + (1-tau) * u1

    def M(u, v):
        """ Modified cost for logarithmic updates. """
        return (-C + tf.transpose(u) + v) / eps

    def squeeze_lse(arg):
        """ Preform squeeze and log sum exp. """
        return tf.squeeze(tf.reduce_logsumexp(arg, axis=1, keepdims=True))

    u, v, err = tf.zeros_like(mu), tf.zeros_like(nu), 0.
    actual_nits = 0

    for _ in range(niter):
        u1 = u
        if unbalanced:
            u = ave(u, lam*(eps*(tf.math.log(mu)-squeeze_lse(M(u, v))+u)))
            v = ave(v, lam*(eps*(tf.math.log(nu)
                            - squeeze_lse(tf.transpose(M(u, v)))+v)))
        else:
            u += eps * (tf.math.log(mu) - squeeze_lse(M(u, v)))
            v += eps * (tf.math.log(nu) - squeeze_lse(tf.transpose(M(u, v))))

        err = tf.reduce_sum(tf.abs(u - u1))
        actual_nits += 1
#        if err/tf.reduce_sum(tf.abs(u)) < thresh:
#            break

#    tf.print(actual_nits, err/tf.reduce_sum(tf.abs(u)))
    return tf.reduce_sum(tf.exp(M(u, v)) * C)


def cost_matrix(x, y, p=2):
    """ Returns the matrix of |x_i - y_j|^p. """
    x_col = tf.expand_dims(x, 1)
    y_row = tf.expand_dims(y, 0)
    return tf.reduce_sum(tf.abs(x_col - y_row)**p, axis=2)


def main():
    """ Basic test of the Sinkhorn loss. """
    import numpy as np
    import matplotlib.pyplot as plt

    n = 1000
    m = 1000
    N = [n, m]

    x = tf.convert_to_tensor(np.random.rand(N[0], 2)-0.5, dtype=tf.float32)
    theta = 2*np.pi*np.random.rand(1, N[1])
    r = 0.8 + 0.2 * np.random.rand(1, N[1])
    y = tf.convert_to_tensor(np.vstack((np.cos(theta)*r, np.sin(theta)*r)).T,
                             dtype=tf.float32)

    def plotp(x, col):
        plt.scatter(x[:, 0], x[:, 1], s=50,
                    edgecolors='k', c=col, linewidths=1)

    mu = tf.random.uniform([n])
    mu /= tf.reduce_sum(mu)
    nu = tf.ones(m)/m

    plt.figure(figsize=(6, 6))
    plotp(x, 'b')
    plotp(y, 'r')

    plt.xlim(np.min(y[:, 0]) - 0.1, np.max(y[:, 0]) + 0.1)
    plt.ylim(np.min(y[:, 1]) - 0.1, np.max(y[:, 1]) + 0.1)
    plt.title('Input marginals')

    eps = tf.constant(0.5)
    niter = 100

    with tf.GradientTape() as tape:
        # l1 = sinkhorn_loss(x, y, eps, mu, nu, niter=5)
        l2 = sinkhorn_normalized(y, y, eps, mu, nu, niter=2)

        # print('Sinkhorn loss: ', l1)
        print('Sinkhorn normalized: ', l2)

        # l1 = sinkhorn_loss(x, y, eps, mu, nu, niter=5)
        l2 = sinkhorn_normalized(y, y, eps, mu, nu, niter=10)

        # print('Sinkhorn loss: ', l1)
        print('Sinkhorn normalized: ', l2)

        # l1 = sinkhorn_loss(x, y, eps, mu, nu, niter=5)
        l2 = sinkhorn_normalized(y, y, eps, mu, nu, niter=100)

        # print('Sinkhorn loss: ', l1)
        print('Sinkhorn normalized: ', l2)

    print(tape.gradient(l2, y))
    # plt.show()


if __name__ == '__main__':
    main()
