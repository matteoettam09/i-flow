import numpy as np
import piecewise
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib.pyplot as plt
import tensorflow as tf

class BijectorFactory:
    def __init__(self):
        self._bijectors = {}

    def register_bijector(self, key, bijector):
        self._bijectors[key] = bijector

    def create(self, key, **kwargs):
        bijector = self._bijectors.get(key)
        if not bijector:
            raise ValueError(key)
        return bijector(**kwargs)

factory = BijectorFactory()
factory.register_bijector('linear', piecewise.PiecewiseLinear)
factory.register_bijector('quadratic', piecewise.PiecewiseQuadratic)
factory.register_bijector('quadratic_const', piecewise.PiecewiseQuadraticConst)

class Integrator():
    def __init__(self, func, ndims, layers=4, mode='quadratic', nbins=25):
        self.func = func
        self.ndims = ndims
        self.mode = mode
        self.nbins = nbins
        self.layers = layers

        self.losses = []
        self.integrals = []
        self.vars = []
        self.global_step = 0

        self.bijectors = []

        arange = np.arange(ndims)
        permute = np.hstack([arange[ndims//2:],arange[:ndims//2]])
        if ndims % 2 != 0:
            permute_odd = np.hstack([arange[ndims//2+1:],arange[:ndims//2+1]])
            odd = True
        else:
            odd = False

        for i in range(layers):
            if not odd:
                self.bijectors.append(factory.create(
                    mode,**{
                        'D': ndims,
                        'd': ndims//2,
                        'nbins': nbins,
                        'layer_id': i,
                        }
                    ))
                self.bijectors.append(tfb.Permute(permutation=permute))
            else:
                if i % 2 == 0:
                    self.bijectors.append(factory.create(
                        mode,**{
                            'D': ndims,
                            'd': ndims//2+1,
                            'nbins': nbins,
                            'layer_id': i,
                            }
                        ))
                    self.bijectors.append(tfb.Permute(permutation=permute))
                else:
                    self.bijectors.append(factory.create(
                        mode,**{
                            'D': ndims,
                            'd': ndims//2,
                            'nbins': nbins,
                            'layer_id': i,
                            }
                        ))
                    self.bijectors.append(tfb.Permute(permutation=permute_odd))

        # Remove the last permute layer
        self.bijectors = tfb.Chain(list(reversed(self.bijectors[:-1]))) 

        self.base_dist = tfd.Uniform(low=ndims*[0.],high=ndims*[1.])
        self.base_dist = tfd.Independent(distribution=self.base_dist,
                reinterpreted_batch_ndims=1,
                )

        self.dist = tfd.TransformedDistribution(
                distribution=self.base_dist,
                bijector=self.bijectors,
                )

    def _loss_fn(self,nsamples):
        x = self.dist.sample(nsamples)
        logq = self.dist.log_prob(x)
        p = self.func(x)
        q = self.dist.prob(x)
        xsec = p/q
        p = p/tf.reduce_mean(xsec)
        mean, var = tf.nn.moments(xsec,axes=[0])
        return tf.reduce_mean(p/q*(tf.log(p)-logq)), mean, var/nsamples

    def make_optimizer(self,learning_rate=1e-4,nsamples=500):
        self.loss, self.integral, self.var = self._loss_fn(nsamples) 
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        self.opt_op = optimizer.apply_gradients(grads)

    def optimize(self,sess,epochs=1000,learning_rate=1e-4,
                 nsamples=500,stopping=1e-4,printout=100):
        for epoch in range(epochs):
            _, np_loss, np_integral, np_var = sess.run([self.opt_op, self.loss, self.integral, self.var])
            self.global_step += 1
            self.losses.append(np_loss)
            self.integrals.append(np_integral)
            self.vars.append(np_var)
            if epoch % printout == 0:
                print("Epoch %4d: loss = %e, average integral = %e, average variance = %e"   
                        %(epoch, np_loss, np.mean(self.integrals), np.mean(self.vars))) 
#            if np.sqrt(np_var)/np_integral < stopping:
#                break

        print("Epoch %4d: loss = %e, average integral = %e, average variance = %e"   
                %(epoch, np_loss, np.mean(self.integrals), np.mean(self.vars))) 

    def _plot(self,axis,labelsize=17,titlesize=20):
        axis.set_xlabel('epoch',fontsize=titlesize)
        axis.tick_params(axis='both',reset=True,which='both',direction='in',size=labelsize)
        return axis
    
    def plot_loss(self,axis,labelsize=17,titlesize=20,start=0):
        axis.plot(self.losses[start:])
        axis.set_ylabel('loss',fontsize=titlesize)
        axis.set_yscale('log')
        axis = self._plot(axis,labelsize,titlesize)
        return axis

    def plot_integral(self,axis,labelsize=17,titlesize=20,start=0):
        axis.plot(self.integrals[start:])
        axis.set_ylabel('integral',fontsize=titlesize)
        axis = self._plot(axis,labelsize,titlesize)
        return axis

    def plot_variance(self,axis,labelsize=17,titlesize=20,start=0):
        axis.plot(self.vars[start:])
        axis.set_ylabel('variance',fontsize=titlesize)
        axis.set_yscale('log')
        axis = self._plot(axis,labelsize,titlesize)
        return axis

    def integrate(self,sess,nsamples=10000):
        x = self.dist.sample(nsamples)
        q = self.dist.prob(x)
        p = self.func(x)
        integral, var = tf.nn.moments(p/q,axes=[0])
        error = tf.sqrt(var/nsamples)

        return sess.run([integral, error])
        

if __name__ == '__main__':
    import tensorflow as tf
    def normalChristina(x):
        return 0.8* tf.exp((-0.5*((x[:,0]-0.5)* (50 *(x[:,0]-0.5) -  15* (x[:,1]-0.5)) + (-15*(x[:,0]-0.5) + 5*(x[:,1]-0.5))* (x[:,1]-0.5)))) + x[:,2]

    integrator = Integrator(normalChristina, 3, mode='quadratic_const')
    integrator.make_optimizer(nsamples=1000)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        integrator.optimize(sess,epochs=5000)
        print(integrator.integrate(sess,10000))

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,5))
    ax1 = integrator.plot_loss(ax1)
    ax2 = integrator.plot_integral(ax2)
    ax3 = integrator.plot_variance(ax3)
    plt.show() 
