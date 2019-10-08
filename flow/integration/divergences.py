""" Implementation of the divergences. """

import tensorflow as tf


class Divergence:
    """ Implement divergence class conatiner. """
    def __init__(self, alpha=None, beta=None):
        self.alpha = alpha
        self.beta = beta
        self.divergences = [x for x in dir(self)
                            if ('__' not in x and 'alpha' not in x
                                and 'beta' not in x)]

    @staticmethod
    def chi2(true, test, logp, logq):
        """ Implement chi2 divergence. """
        del logp, logq
        return tf.reduce_mean(input_tensor=(tf.stop_gradient(true) - test)**2
                              / test / tf.stop_gradient(test))

    @staticmethod
    def chi2sym(true, test, logp, logq):
        """ Implement symmetric chi2 divergence. """
        del logp, logq
        return 0.5*(tf.reduce_mean(input_tensor=(tf.stop_gradient(true) - test)**2
                              / test / tf.stop_gradient(test))
                    + tf.reduce_mean(input_tensor=(tf.stop_gradient(true) - test)**2
                              /tf.stop_gradient(true+1e-16)  / tf.stop_gradient(test)))
    
    @staticmethod
    def chi2sym2(true, test, logp, logq):
        """ Implement chi2 divergence with different normalization. """
        del logp, logq
        return tf.reduce_mean(input_tensor=(tf.stop_gradient(true) - test)**2
                              / ((test+tf.stop_gradient(true))/2.) / tf.stop_gradient(test))
    
    # pylint: disable=invalid-name
    @staticmethod
    def kl(true, test, logp, logq):
        """ Implement kl divergence. """
        return tf.reduce_mean(input_tensor=tf.stop_gradient(true/test)
                              * (tf.stop_gradient(logp) - logq))
    # pylint: enable=invalid-name

    @staticmethod
    def hellinger(true, test, logp, logq):
        """ Implement Hellinger divergence. """
        del logp, logq
        return tf.reduce_mean(
            input_tensor=(2.0*(tf.stop_gradient(tf.math.sqrt(true))
                               - tf.math.sqrt(test))**2
                          / tf.stop_gradient(test)))

    @staticmethod
    def jeffreys(true, test, logp, logq):
        """ Implement Jeffreys divergence. """
        return tf.reduce_mean(
            input_tensor=((tf.stop_gradient(true) - test)
                          * (tf.stop_gradient(logp) - logq)
                          / tf.stop_gradient(test)))

    def chernoff(self, true, test, logp, logq):
        """ Implement Chernoff divergence. """
        del logp, logq
        if self.alpha is None:
            raise ValueError('Must give an alpha value to use Chernoff '
                             'Divergence.')
        if not 0 < self.alpha < 1:
            raise ValueError('Alpha must be between 0 and 1.')

        return (4.0 / (1-self.alpha**2)*(1 - tf.reduce_mean(
            input_tensor=(tf.stop_gradient(tf.pow(true,
                                                  (1.0-self.alpha)/2.0))
                          * tf.pow(test, (1.0+self.alpha)/2.0)
                          / tf.stop_gradient(test)))))

    @staticmethod
    def exponential(true, test, logp, logq):
        """ Implement Expoential divergence. """
        return tf.reduce_mean(
            input_tensor=tf.stop_gradient(true/test)*(
                tf.stop_gradient(logp) - logq)**2)

    def ab_product(self, true, test, logp, logq):
        """ Implement (alpha, beta)-product divergence. """
        del logp, logq
        if self.alpha is None:
            raise ValueError('Must give an alpha value to use '
                             '(alpha, beta)-product Divergence.')
        if not 0 < self.alpha < 1:
            raise ValueError('Alpha must be between 0 and 1.')

        if self.beta is None:
            raise ValueError('Must give an beta value to use '
                             '(alpha, beta)-product Divergence.')
        if not 0 < self.beta < 1:
            raise ValueError('Beta must be between 0 and 1.')

        return tf.reduce_mean(
            input_tensor=(2.0/((1-self.alpha)*(1-self.beta))
                          * (1-tf.pow(test/tf.stop_gradient(true),
                                      (1-self.alpha)/2.0))
                          * (1-tf.pow(test/tf.stop_gradient(true),
                                      (1-self.beta)/2.0))
                          * tf.stop_gradient(true/test)))

    # pylint: disable=invalid-name
    @staticmethod
    def js(true, test, logp, logq):
        """ Implement Jensen–Shannon divergence. """
        logm = tf.math.log(0.5*(test+tf.stop_gradient(true)))
        return tf.reduce_mean(input_tensor=(
            tf.stop_gradient(0.5/test) * ((tf.stop_gradient(true)
                                           * (tf.stop_gradient(logp)-logm))
                                          + (test * (logq-logm)))))
    # pylint: enable=invalid-name

    def __call__(self, name):
        func = getattr(self, name, None)
        if func is not None:
            return func
        raise NotImplementedError('The requested loss function {} '
                                  'is not implemented. Allowed '
                                  'options are {}.'.format(
                                      name, self.divergences))
