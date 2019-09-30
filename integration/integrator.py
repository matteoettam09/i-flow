import numpy as np
import piecewise
import piecewiseUnet_one_hot
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import matplotlib.pyplot as plt
import tensorflow as tf
import corner

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
factory.register_bijector('linear_unet', piecewiseUnet_one_hot.PiecewiseLinearUNet)
factory.register_bijector('quadratic_unet', piecewiseUnet_one_hot.PiecewiseQuadraticUNet)
factory.register_bijector('quadratic_const_unet', piecewiseUnet_one_hot.PiecewiseQuadraticConstUNet)

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
        permute = np.hstack([arange[1:],arange[0]])
        for i in range(ndims):
            self.bijectors.append(factory.create(
                    mode,**{
                        'D': ndims,
                        'd': ndims//2,
                        'nbins': nbins,
                        'layer_id': i,
                        }
                    ))
            self.bijectors.append(tfb.Permute(permutation=permute))
#        if ndims % 2 != 0:
#            permute_odd = np.hstack([arange[ndims//2+1:],arange[:ndims//2+1]])
#            odd = True
#        else:
#            odd = False
#
#        for i in range(layers):
#            if not odd:
#                self.bijectors.append(factory.create(
#                    mode,**{
#                        'D': ndims,
#                        'd': ndims//2,
#                        'nbins': nbins,
#                        'layer_id': i,
#                        }
#                    ))
#                self.bijectors.append(tfb.Permute(permutation=permute))
#            else:
#                if i % 2 == 0:
#                    self.bijectors.append(factory.create(
#                        mode,**{
#                            'D': ndims,
#                            'd': ndims//2+1,
#                            'nbins': nbins,
#                            'layer_id': i,
#                            }
#                        ))
#                    self.bijectors.append(tfb.Permute(permutation=permute))
#                else:
#                    self.bijectors.append(factory.create(
#                        mode,**{
#                            'D': ndims,
#                            'd': ndims//2,
#                            'nbins': nbins,
#                            'layer_id': i,
#                            }
#                        ))
#                    self.bijectors.append(tfb.Permute(permutation=permute_odd))

        # Remove the last permute layer
        self.bijectors = tfb.Chain(list(reversed(self.bijectors))) 

        self.base_dist = tfd.Uniform(low=ndims*[0.],high=ndims*[1.])
        self.base_dist = tfd.Independent(distribution=self.base_dist,
                reinterpreted_batch_ndims=1,
                )

        self.dist = tfd.TransformedDistribution(
                distribution=self.base_dist,
                bijector=self.bijectors,
                )

        self.saver = tf.train.Saver()

    def _loss_fn(self,nsamples):
        x = self.dist.sample(nsamples)
        logq = self.dist.log_prob(x)
        p = self.func(x)
        q = self.dist.prob(x)
        xsec = p/q
        p = p/tf.reduce_mean(xsec)
        mean, var = tf.nn.moments(xsec,axes=[0])
        return tf.reduce_mean(p/q*(tf.log(p)-logq)), mean, var/nsamples, x, p, q

    def make_optimizer(self,learning_rate=1e-4,nsamples=500):
        self.loss, self.integral, self.var, self.x, self.p, self.q = self._loss_fn(nsamples) 
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        self.opt_op = optimizer.apply_gradients(grads)

    def optimize(self,sess,epochs=1000,learning_rate=1e-4,
                 nsamples=500,stopping=1e-4,printout=100,
                 profiler=None,options=None):

        for epoch in range(epochs):
            if profiler is not None:
                run_metadata = tf.RunMetadata()
            else:
                run_metadata = None
            _, np_loss, np_integral, np_var, xpts, ppts, qpts = sess.run([self.opt_op, self.loss, self.integral, self.var, self.x, self.p, self.q],options=options,run_metadata=run_metadata)
            if profiler is not None:
                profiler.add_step(epoch, run_metadata)
            self.global_step += 1
            self.losses.append(np_loss)
            self.integrals.append(np_integral)
            self.vars.append(np_var)
            if epoch % printout == 0:#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

                print("Epoch %4d: loss = %e, average integral = %e, average variance = %e"   
                        %(epoch, np_loss, np.mean(self.integrals), np.mean(self.vars))) 
                figure = corner.corner(xpts, labels=[r'$x_{}$'.format(i) for i in range(self.ndims)], show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
                plt.savefig('fig_{:04d}.pdf'.format(epoch))
                plt.close()
#            if np.sqrt(np_var)/np_integral < stopping:
#                break

        print("Epoch %4d: loss = %e, average integral = %e, average variance = %e"   
                %(epoch, np_loss, np.mean(self.integrals), np.mean(self.vars))) 
        figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$',r'$x_6$',r'$x_7$',r'$x_8$'], show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('fig_{:04d}.pdf'.format(epoch))
        plt.close()

    def save(self,sess,name):
        save_path = self.saver.save(sess, name)
        print("Model saved at: {}".format(save_path))

    def load(self,sess,name):
        self.saver.restore(sess, name)
        print("Model resotred")

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

        np_int, np_error, xpts, qpts = sess.run([integral,error,x,q])
        figure = corner.corner(xpts, labels=[r'$x_{}$'.format(i) for i in range(self.ndims)], weights = qpts, show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('xsec_final.pdf')
        plt.close()

        return np_int, np_error

    def acceptance(self,sess,nsamples=10000):
        x = self.dist.sample(nsamples)
        q = self.dist.prob(x)
        p = self.func(x)
        integral, var = tf.nn.moments(p/q,axes=[0])
        p = p/integral
        r = tf.minimum(p/q,1.0)

        np_r = sess.run(r)

        plt.hist(np_r)
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('acceptance.pdf')
        plt.close()

        return
        

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
