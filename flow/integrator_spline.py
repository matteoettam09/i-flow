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
            p = self._func(x)
            q = self.dist.prob(x)
            xsec = p/q
            mean, var = tf.nn.moments(x=xsec, axes=[0])
            p = p/mean
            loss = tf.reduce_mean(input_tensor=tf.stop_gradient(p/q)*(tf.stop_gradient(tf.math.log(p))-logq))
           
        grads = tape.gradient(loss, self.dist.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, dist.trainable_variables))

        self.losses.append(loss.numpy())

        if integral:
            self.integrals.append(mean.numpy())
            self.vars.append(var.numpy()/nsamples)
            return loss.numpy(), mean.numpy(), np.sqrt(var.numpy()/nsamples)

        return loss.numpy()

    def sample(self, nsamples):
        return self.dist.sample(nsamples)

    def integrate(self, nsamples):
        x = self.dist.sample(nsamples)
        q = self.dist.prob(x)
        p = self._func(x)

        return tf.nn.moments(x=p/q, axes=[0])

    def acceptance(self, nsamples):
        x = self.dist.sample(nsamples)
        q = self.dist.prob(x)
        p = self._func(x)

        return p/q


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    class CosineAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, total_epochs, eta_min=0):
            self.base_lr = base_lr
            self.total_epochs = total_epochs
            self.eta_min = eta_min
            
        def __call__(self,step):
            frac_epochs = step / self.total_epochs
            return self.eta_min + (self.base_lr - self.eta_min) \
                    * (1 + np.cos(np.pi * frac_epochs)) / 2


    def build_dense(in_features, out_features):
        invals = tf.keras.layers.Input(in_features)
#        h_widths = tf.keras.layers.Dense(16)(invals)
#        h_heights = tf.keras.layers.Dense(16)(invals)
#        h_derivs = tf.keras.layers.Dense(16)(invals)
#        h_widths = tf.keras.layers.Dense((out_features-1)/3)(h_widths)
#        h_heights = tf.keras.layers.Dense((out_features-1)/3)(h_heights)
#        h_derivs = tf.keras.layers.Dense((out_features-1)/3+2)(h_derivs)
#        outputs = tf.keras.layers.Concatenate()([h_widths, h_heights, h_derivs])
        h = tf.keras.layers.Dense(16)(invals)
        h = tf.keras.layers.Dense(32)(h)
        h = tf.keras.layers.Dense(64)(h)
        h = tf.keras.layers.Dense(128)(h)
        outputs = tf.keras.layers.Dense(out_features)(h)
        model = tf.keras.models.Model(invals,outputs)
        model.summary()
        return model

    def func(x):
        return tf.reduce_prod(x,axis=-1)

    def camel(x):
        return (tf.exp(-1./0.004*((x[:,0]-0.25)**2+(x[:,1]-0.25)**2))
               +tf.exp(-1./0.004*((x[:,0]-0.75)**2+(x[:,1]-0.75)**2)))
    
    ndims = 2
    epochs = int(500)
    
    bijectors = []
    masks = [[x % 2 for x in range(1,ndims+1)],[x % 2 for x in range(0,ndims)],[1 if x < ndims/2 else 0 for x in range(0,ndims)],[0 if x < ndims/2 else 1 for x in range(0,ndims)]]
    bijectors.append(couplings.PiecewiseRationalQuadratic([1,0],build_dense,num_bins=32))
    bijectors.append(couplings.PiecewiseRationalQuadratic([0,1],build_dense,num_bins=32))
    bijectors.append(couplings.PiecewiseRationalQuadratic([1,0],build_dense,num_bins=32))
    bijectors.append(couplings.PiecewiseRationalQuadratic([0,1],build_dense,num_bins=32))
    
    bijectors = tfb.Chain(list(reversed(bijectors)))
    
    base_dist = tfd.Uniform(low=ndims*[0.], high=ndims*[1.])
    base_dist = tfd.Independent(distribution=base_dist,
                                reinterpreted_batch_ndims=1,
                                )
    dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=bijectors,
    )

    initial_learning_rate = 5e-3
    lr_schedule = CosineAnnealing(initial_learning_rate,epochs)
    
    optimizer = tf.keras.optimizers.Adam(lr_schedule, clipvalue = 5.0)#lr_schedule)
    
    integrator = Integrator(camel, dist, optimizer)
    try:
        for epoch in range(epochs):
            #if epoch % 1 == 0:
            #    samples = integrator.sample(100000)
            #    plt.scatter(samples[:,0],samples[:,1],s=0.1)
            #    plt.savefig('fig_{:04d}.png'.format(epoch))
            #    plt.close()
            loss = integrator.train_one_step(5000,integral=True)
            if epoch % 10 == 0:
                print(epoch, loss)
    except KeyboardInterrupt:
        pass

#    samples = integrator.sample(100000)
#    plt.scatter(samples[:,0],samples[:,1],s=0.1)
#    plt.savefig('fig_{:04d}.png'.format(epochs))
#    plt.close()

    weights = integrator.acceptance(100000).numpy()
    average = np.mean(weights)
    max_wgt = np.max(weights)

    print(average/max_wgt)

    plt.hist(weights,bins=100,range=[np.min(weights),np.max(weights)])
    plt.axvline(average,linestyle='--')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    
    plt.plot(integrator.losses)
    plt.yscale('log')
    plt.show()
