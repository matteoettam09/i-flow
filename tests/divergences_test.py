""" Test divergence implementation. """

# pylint: disable=invalid-name, redefined-outer-name

import pytest

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from flow.integration import divergences

tfd = tfp.distributions

MU_1, MU_2 = 0., 1.
SIGMA_1, SIGMA_2 = 1., 2.
NSAMPLES = 10000


@pytest.fixture
def divergence(scope='module'):  # pylint: disable=unused-argument
    """ Build divergence class to be used by all codes. """
    return divergences.Divergence()


@pytest.fixture
def distributions(scope='module'):  # pylint: disable=unused-argument
    """ Build the distributions once as fixtures. """
    dist_p = tfd.Normal(loc=MU_1, scale=SIGMA_1)
    dist_q = tfd.Normal(loc=MU_2, scale=SIGMA_2)

    sample = dist_q.sample(NSAMPLES)

    prob_p = dist_p.prob(sample)
    prob_q = dist_q.prob(sample)

    logp = dist_p.log_prob(sample)
    logq = dist_q.log_prob(sample)

    return prob_p, prob_q, logp, logq


def test_chi2_divergence(divergence, distributions):
    """ Test the chi2 divergence agains gaussians. """

    loss = divergence('chi2')(*distributions)
    expected = (tf.exp(-(MU_1-MU_2)**2/(SIGMA_1**2-2*SIGMA_2**2))*SIGMA_2**2
                / (SIGMA_1*tf.sqrt(-SIGMA_1**2+2*SIGMA_2**2)) - 1)

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_kl_divergence(divergence, distributions):
    """ Test the kl divergence against gaussians. """

    loss = divergence('kl')(*distributions)
    expected = (tf.math.log(SIGMA_2/SIGMA_1)
                + (SIGMA_1**2+(MU_1 - MU_2)**2)/(2*SIGMA_2**2)
                - 0.5)

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_hellinger_divergence(divergence, distributions):
    """ Test the Hellinger divergence against gaussians. """

    loss = divergence('hellinger')(*distributions)
    expected = (4.0-4*tf.sqrt((2.0*SIGMA_1*SIGMA_2)/(SIGMA_1**2 + SIGMA_2**2))
                * tf.exp(-0.25*(MU_1-MU_2)**2/(SIGMA_1**2 + SIGMA_2**2)))

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_jeffreys_divergence(divergence, distributions):
    """ Test the Jeffreys divergence against gaussians. """

    loss = divergence('jeffreys')(*distributions)
    expected = (((MU_1-MU_2)**2*SIGMA_1**2+SIGMA_1**4
                 + ((MU_1-MU_2)**2-2*SIGMA_1**2)*SIGMA_2**2+SIGMA_2**4)
                / (2*SIGMA_1**2*SIGMA_2**2))

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_chernoff_divergence(divergence, distributions):
    """ Test the Chernoff's alpha-divergence against gaussians. """

    alpha = 0.5
    divergence.alpha = alpha
    loss = divergence('chernoff')(*distributions)
    expected = 4.0/(1-alpha**2)*(
        1-(tf.sqrt(2.0)*tf.exp((-1+alpha**2)*(MU_1-MU_2)**2
                               / (4*(1+alpha)*SIGMA_1**2
                                  - 4*(-1+alpha)*SIGMA_2**2))
           * (SIGMA_1/SIGMA_2)**(alpha/2.0)
           * tf.sqrt((SIGMA_1*SIGMA_2)
                     / ((1+alpha)*SIGMA_1**2-(-1+alpha)*SIGMA_2**2))))

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_exponential_divergence(divergence, distributions):
    """ Test the exponential divergence against gaussians. """

    loss = divergence('exponential')(*distributions)
    expected = 1.0/(4.0*SIGMA_2**4)*(
        MU_1**4-4*MU_1**3*MU_2+6*MU_1**2*MU_2**2-4*MU_1*MU_2**3+MU_2**4
        + 6*MU_1**2*SIGMA_1**2-12*MU_1*MU_2*SIGMA_1**2+6*MU_2**2*SIGMA_1**2
        + 3*SIGMA_1**4-2*((MU_1-MU_2)**2+3*SIGMA_1**2)*SIGMA_2**2
        + 3*SIGMA_2**4-4*SIGMA_2**2*tf.math.log(SIGMA_1/SIGMA_2)*(
            (MU_1-MU_2)**2+(SIGMA_1-SIGMA_2)*(SIGMA_1+SIGMA_2)
            + SIGMA_2**2*tf.math.log(SIGMA_2/SIGMA_1)))

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_ab_product_divergence(divergence, distributions):
    """ Test the (alpha, beta)-product divergence against gaussians. """

    alpha = 0.5
    beta = 0.5
    divergence.alpha = alpha
    divergence.beta = beta
    loss = divergence('ab_product')(*distributions)

    def prefactor(alpha, beta):
        return (tf.pow(2.0*np.pi, 0.5*(1-alpha-beta))
                * tf.pow(SIGMA_1, 1-beta)
                * tf.pow(SIGMA_2, 1-alpha)
                * tf.sqrt(1.0/(alpha*SIGMA_1**2 + beta*SIGMA_2**2)))

    def exponent(alpha, beta):
        return tf.exp(-(MU_1 - MU_2)**2
                      / (2.0*(SIGMA_1**2/beta + SIGMA_2**2/alpha)))

    expected = 2.0/((1-alpha)*(1-beta)) \
        * (1.0
           - prefactor((1-alpha)/2., 1-(1-alpha)/2.)
           * exponent((1-alpha)/2., 1-(1-alpha)/2.)
           - prefactor((1-beta)/2., 1-(1-beta)/2.)
           * exponent((1-beta)/2., 1-(1-beta)/2.)
           + prefactor(1-alpha/2.-beta/2.,
                       (alpha+beta)/2.)
           * exponent(1-alpha/2.-beta/2.,
                      (alpha+beta)/2.))

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))


def test_jensen_shannon(divergence, distributions):
    """ Test the Jensen-Shannon divergence against gaussians. """

    loss = divergence('js')(*distributions)

    prob_m = 0.5*(distributions[0] + distributions[1])
    log_m = tf.math.log(prob_m)
    expected = 0.5*(divergence('kl')(distributions[0], distributions[1],
                                     distributions[2], log_m)
                    + divergence('kl')(distributions[1], distributions[1],
                                       distributions[3], log_m))

    assert abs(loss - expected) <= 3*loss/tf.sqrt(float(NSAMPLES))
