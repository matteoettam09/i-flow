""" Implement the Sinkhorn loss.

    Modified from the pyTorch implementation found here,
    https://github.com/gpeyre/SinkhornAutoDiff/tree/generalization
    into Tensorflow.
"""

import tensorflow as tf


def dist(pts_x, pts_y):
    """ Calculate Euclidean distance squared between two vectors. """
    pts_xx = tf.einsum('ij,ij->i', pts_x, pts_x)[:, tf.newaxis]
    pts_yy = tf.einsum('ij,ij->i', pts_y, pts_y)[tf.newaxis, :]

    distances = pts_xx + pts_yy - 2*tf.matmul(pts_x, tf.transpose(pts_y))
    return tf.maximum(distances, 0)


def sinkhorn2(wgt_a, wgt_b, mat, reg, niter):
    """ Preform Sinkhorn iterations to converge on the loss. """
    wgt_a = tf.convert_to_tensor(wgt_a, dtype=tf.float64)
    wgt_b = tf.convert_to_tensor(wgt_b, dtype=tf.float64)
    mat = tf.convert_to_tensor(mat, dtype=tf.float64)

    dim_a = len(wgt_a)
    dim_b = len(wgt_b)
    vec_u = tf.ones((dim_a, 1), dtype=tf.float64) / dim_a
    vec_v = tf.ones((dim_b, 1), dtype=tf.float64) / dim_b

    mat_k = tf.zeros_like(mat, dtype=tf.float64)
    mat_k = tf.divide(mat, -reg)
    mat_k = tf.exp(mat_k)

    tmp2 = tf.zeros_like(wgt_b, dtype=tf.float64)

    mat_kp = tf.reshape((1 / wgt_a), (-1, 1)) * mat_k
    cpt = 0
    err = 1
    while (err > 1e-6 and cpt < niter):
        k_transpose_u = tf.matmul(tf.transpose(mat_k), vec_u)
        vec_v = wgt_b/k_transpose_u
        vec_u = 1. / tf.matmul(mat_kp, vec_v)

        if cpt % 10 == 0:
            tmp2 = tf.einsum('ik,ij,jk->jk', vec_u, mat_k, vec_v)
            err = tf.norm(tmp2 - wgt_b)

        cpt += 1

    return tf.einsum('ik,ij,jk,ij->k', vec_u, mat_k, vec_v, mat)


def empirical_sinkhorn2(x_s, x_t, reg, wgt_a, wgt_b, niter=100):
    """ Calculate the Sinkhorn loss. """
    x_s = tf.convert_to_tensor(x_s, dtype=tf.float64)
    x_t = tf.convert_to_tensor(x_t, dtype=tf.float64)

    mat = dist(x_s, x_t)

    return sinkhorn2(wgt_a, wgt_b, mat, reg, niter)


@tf.function
def sinkhorn_loss(x_s, x_t, reg, wgt_a, wgt_b, niter):
    """ Calculate the Sinkhorn divergence. """
    if len(wgt_a.shape) != 2:
        wgt_a = wgt_a[:, tf.newaxis]

    if len(wgt_b.shape) != 2:
        wgt_b = wgt_b[:, tf.newaxis]

    sinkhorn_ab = empirical_sinkhorn2(x_s, x_t, reg, wgt_a, wgt_b, niter)
    sinkhorn_a = empirical_sinkhorn2(x_s, x_s, reg, wgt_a, wgt_a, niter)
    sinkhorn_b = empirical_sinkhorn2(x_t, x_t, reg, wgt_b, wgt_b, niter)

    print(sinkhorn_ab - 1/2 * (sinkhorn_a + sinkhorn_b))

    return (sinkhorn_ab - 1/2 * (sinkhorn_a + sinkhorn_b))[0]


# @tf.function
# def sinkhorn_normalized(true, test, eps, mu, nu, niter):
#     """ Computes Sinkhorn divergence. """
#     print('here', true, test, eps, mu, nu)
#     w_xy = sinkhorn_loss(true, test, eps, mu, nu, niter)
#     w_xx = sinkhorn_loss(true, true, eps, mu, mu, niter)
#     w_yy = sinkhorn_loss(test, test, eps, nu, nu, niter)
#     print(w_xy, w_xx, w_yy)

#     return w_xy - 0.5 * w_xx - 0.5 * w_yy


# @tf.function
# def sinkhorn_loss(true, test, eps, mu, nu, niter=100,
#                   acc=1e-3, unbalanced=False):
#     """ Calculate the Sinkhorn loss. """
#     C = cost_matrix(true, test)

#     tau = -0.8
#     thresh = acc

#     if unbalanced:
#         rho = (0.5)**2
#         lam = rho / (rho + eps)

#     def ave(u, u1):
#         """ Barycenter subroutine. """
#         return tau * u + (1-tau) * u1

#     def M(u, v):
#         """ Modified cost for logarithmic updates. """
#         return (-C + tf.transpose(u) + v) / eps

#     def squeeze_lse(arg):
#         """ Preform squeeze and log sum exp. """
#         return tf.squeeze(tf.reduce_logsumexp(arg, axis=1, keepdims=True))

#     u, v, err = tf.zeros_like(mu), tf.zeros_like(nu), 0.
#     actual_nits = 0

#     for _ in range(niter):
#         u1 = u
#         if unbalanced:
#             u = ave(u, lam*(eps*(tf.math.log(mu)-squeeze_lse(M(u, v))+u)))
#             v = ave(v, lam*(eps*(tf.math.log(nu)
#                             - squeeze_lse(tf.transpose(M(u, v)))+v)))
#         else:
#             u += eps * (tf.math.log(mu) - squeeze_lse(M(u, v)))
#             v += eps * (tf.math.log(nu) - squeeze_lse(tf.transpose(M(u, v))))

#         err = tf.reduce_sum(tf.abs(u - u1))
#         actual_nits += 1
# #        if err/tf.reduce_sum(tf.abs(u)) < thresh:
# #            break

# #    tf.print(actual_nits, err/tf.reduce_sum(tf.abs(u)))
#     return tf.reduce_sum(tf.exp(M(u, v)) * C)


# def cost_matrix(x, y, p=2):
#     """ Returns the matrix of |x_i - y_j|^p. """
#     x_col = tf.expand_dims(x, 1)
#     y_row = tf.expand_dims(y, 0)
#     return tf.reduce_sum(tf.abs(x_col - y_row)**p, axis=2)


# def main():
#     """ Basic test of the Sinkhorn loss. """
#     import numpy as np
#     import matplotlib.pyplot as plt

#     n = 1000
#     m = 1000
#     N = [n, m]

#     x = tf.convert_to_tensor(np.random.rand(N[0], 2)-0.5, dtype=tf.float32)
#     theta = 2*np.pi*np.random.rand(1, N[1])
#     r = 0.8 + 0.2 * np.random.rand(1, N[1])
#     y = tf.convert_to_tensor(np.vstack((np.cos(theta)*r, np.sin(theta)*r)).T,
#                              dtype=tf.float32)

#     def plotp(x, col):
#         plt.scatter(x[:, 0], x[:, 1], s=50,
#                     edgecolors='k', c=col, linewidths=1)

#     mu = tf.random.uniform([n])
#     mu /= tf.reduce_sum(mu)
#     nu = tf.ones(m)/m

#     plt.figure(figsize=(6, 6))
#     plotp(x, 'b')
#     plotp(y, 'r')

#     plt.xlim(np.min(y[:, 0]) - 0.1, np.max(y[:, 0]) + 0.1)
#     plt.ylim(np.min(y[:, 1]) - 0.1, np.max(y[:, 1]) + 0.1)
#     plt.title('Input marginals')

#     eps = tf.constant(0.5)
#     niter = 100

#     with tf.GradientTape() as tape:
#         # l1 = sinkhorn_loss(x, y, eps, mu, nu, niter=5)
#         l2 = sinkhorn_normalized(y, y, eps, mu, nu, niter=2)

#         # print('Sinkhorn loss: ', l1)
#         print('Sinkhorn normalized: ', l2)

#         # l1 = sinkhorn_loss(x, y, eps, mu, nu, niter=5)
#         l2 = sinkhorn_normalized(y, y, eps, mu, nu, niter=10)

#         # print('Sinkhorn loss: ', l1)
#         print('Sinkhorn normalized: ', l2)

#         # l1 = sinkhorn_loss(x, y, eps, mu, nu, niter=5)
#         l2 = sinkhorn_normalized(y, y, eps, mu, nu, niter=100)

#         # print('Sinkhorn loss: ', l1)
#         print('Sinkhorn normalized: ', l2)

#     print(tape.gradient(l2, y))
#     # plt.show()


# if __name__ == '__main__':
#     main()
