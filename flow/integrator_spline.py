import couplings
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

def ewma(data, window):
    """
    Function to caluclate the Exponentially weighted moving average.

    Args:
        data (np.ndarray, float64): An array of data for the average to be calculated with.
        window (int64): The decay window.

    Returns:
        int64: The EWMA for the last point in the data array
    """
    if len(data) <= window:
        return data[-1]

    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(data, weights, mode='full')[:len(data)]
    a[:window] = a[window]
    return a[-1]

class Integrator():

    def __init__(self, func, dist, optimizer, **kwargs):
        self._func = func
        
        self.losses = []
        self.integrals = []
        self.vars = []
        self.global_step = 0

        self.dist = dist

        self.optimizer = optimizer

    def train_one_step(self, nsamples, integral=False):
        with tf.GradientTape() as tape:
            x = tf.stop_gradient(self.dist.sample(nsamples))
            logq = self.dist.log_prob(x)
            p = tf.stop_gradient(func(x))
            q = self.dist.prob(x)
            xsec = p/q
            mean, var = tf.nn.moments(x=xsec, axes=[0])
            p = p/mean
            loss = tf.reduce_mean(input_tensor=p/q*(tf.math.log(p)-logq))
           
        grads = tape.gradient(loss, self.dist.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, dist.trainable_variables))

        self.losses.append(loss.numpy())

        if integral:
            self.integrals.append(mean.numpy())
            self.vars.append((var/nsamples).numpy())
            return loss.numpy(), mean.numpy(), (var/nsamples).numpy()

        return loss.numpy()

if __name__ == '__main__':

    def build_dense(in_features, out_features):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(in_features))
        model.add(tf.keras.layers.Dense(128,activation='relu'))
        model.add(tf.keras.layers.Dense(128,activation='relu'))
        model.add(tf.keras.layers.Dense(128,activation='relu'))
        model.add(tf.keras.layers.Dense(out_features))
        model.summary()
        model.build()
        return model

    def func(x):
        return tf.reduce_prod(x,axis=-1)
    
    ndims = 4
    
    bijectors = []
    masks = [[x % 2 for x in range(1,ndims+1)],[x % 2 for x in range(0,ndims)],[0 if x < ndims/2 else 1 for x in range(0,ndims)],[1 if x < ndims/2 else 0 for x in range(0,ndims)]]
    for mask in masks:
        layer = couplings.PiecewiseRationalQuadratic(mask,build_dense)
        bijectors.append(layer)
    
    bijectors = tfb.Chain(list(bijectors))
    
    base_dist = tfd.Uniform(low=ndims*[0.], high=ndims*[1.])
    base_dist = tfd.Independent(distribution=base_dist,
                                reinterpreted_batch_ndims=1,
                                )
    dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=bijectors,
    )
    
    optimizer = tf.keras.optimizers.Adam(1e-2)
    
    integrator = Integrator(func, dist, optimizer)
    for epoch in range(200):
        loss = integrator.train_one_step(1000,integral=False)
        if epoch % 10 == 0:
            print(epoch, loss)
    print(integrator.train_one_step(1000,integral=True))
    
    import matplotlib.pyplot as plt
    
    plt.plot(integrator.losses)
    plt.yscale('log')
    plt.show()
