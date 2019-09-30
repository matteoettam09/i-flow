import numpy as np
import piecewiseUnet_channel
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
factory.register_bijector('linear_channel', piecewiseUnet_channel.PiecewiseLinear)
factory.register_bijector('quadratic_channel', piecewiseUnet_channel.PiecewiseQuadratic)
factory.register_bijector('quadraticConst_channel', piecewiseUnet_channel.PiecewiseQuadraticConst)
class Integrator():
    def __init__(self, func, ndims, nchannels, layers=4, mode='linear_channel', nbins=32):
        self.func = func
        self.ndims = ndims
        self.mode = mode
        self.nbins = nbins
        self.nchannels = nchannels
        self.layers = layers
        
        self.losses = []
        self.integrals = []
        self.vars = []
        self.global_step = 0

        self.bijectors = []

        arange = np.arange(ndims)
        permute = np.hstack([arange[1:],arange[0]])
        #self.bijectors.append(piecewiseUnet_channel.Permute(permutation=np.array([4,2,1,3,0])))
        for i in range(ndims):
            self.bijectors.append(factory.create(
                    mode,**{
                        'D': ndims,
                        'd': ndims//2 + ndims % 2, #ndims-1,
                        'nbins': nbins,
                        'nchannels': nchannels,
                        'layer_id': i,
                        'name':"L"
                        }
                    ))
            self.bijectors.append(piecewiseUnet_channel.Permute(permutation=permute,name="P"))
            #self.bijectors.append(piecewiseUnet_channel.Permute(permutation=np.array([4,0,1,2,3]),name="P"))
        #self.bijectors.append(piecewiseUnet_channel.Permute(permutation=np.array([4,2,1,3,0])))
        self.bijectors = tfb.Chain(list(reversed(self.bijectors))) 

        self.base_dist = tfd.Uniform(low=ndims*[0.],high=ndims*[1.],name="UnF")
        self.base_dist = tfd.Independent(distribution=self.base_dist,
                reinterpreted_batch_ndims=1,
                )

        self.dist = tfd.TransformedDistribution(
                distribution=self.base_dist,
                bijector=self.bijectors,
                )
        self.saver = tf.train.Saver()
        

    def save(self,sess,name):
        save_path = self.saver.save(sess, name)
        print("Model saved at: {}".format(save_path))

    def load(self,sess,name):
        self.saver.restore(sess, name)
        print("Model restored")

        
    def _loss_fn(self,nsamples,sess):
        ran = tf.cast(tf.floor(tf.random.uniform((nsamples,),minval=0., maxval=self.nchannels,dtype=tf.dtypes.float32)),tf.dtypes.int32)
        ran_np=ran.eval(session=sess)

        randic = {"channel": ran_np.tolist()}
        layerdic = {"L":randic}
        kwargs={"bijector_kwargs":layerdic}
        #x = tf.stop_gradient(self.dist.sample(nsamples,**kwargs))
        x = self.dist.sample(nsamples,**kwargs)
        logq = self.dist.log_prob(x,**kwargs)
        p = self.func(x,randic)
        q = self.dist.prob(x,**kwargs)
        xsec = p/q
        
        #xs = tf.stop_gradient(self.dist.sample(nsamples,**kwargs))
        #xs = self.dist.sample(nsamples,**kwargs)
        #xsec = self.func(xs,randic)/self.dist.prob(xs,**kwargs)

        p = p/tf.reduce_mean(xsec)
        mean, var = tf.nn.moments(xsec,axes=[0])
        # KL divergence:
        return tf.reduce_mean((p/q)*(tf.log(p)-logq)), mean, var/nsamples, x, p, q
        # KL divergence with acceptance:
        #return tf.reduce_mean((p/q)*(tf.log(p)-logq)) + 0.01*(1.-tf.reduce_mean(p/q)/tf.reduce_max(p/q)), mean, var/nsamples, x, p, q

        # chi^2 divergence:
        #return tf.reduce_mean(((p-q)/q)**2), mean, var/nsamples, x, p, q

    def make_optimizer(self,session,learning_rate=1e-4,nsamples=500):
        self.loss, self.integral, self.var, self.x, self.p, self.q = self._loss_fn(nsamples,session) 
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(self.loss)
        self.opt_op = optimizer.apply_gradients(grads)

        
    def optimize(self,sess,epochs=1000,learning_rate=1e-4,
                 nsamples=500,stopping=1e-4,printout=100):
        self.sess = sess
        floating_av = np.array([])
        floating_int = np.array([])
        min_loss = 1e3
        for epoch in range(epochs):
            _, np_loss, np_integral, np_var, xpts, ppts, qpts = sess.run([self.opt_op, self.loss, self.integral, self.var, self.x, self.p, self.q])
            self.global_step += 1
            self.losses.append(np_loss)
            self.integrals.append(np_integral)
            self.vars.append(np_var)
            if epoch // 10 == 0:
                floating_int=np.append(floating_av,np_integral)
                floating_av=np.append(floating_av,np_var)
            else:
                floating_int[epoch % 10] = np_integral
                floating_av[epoch % 10] = np_var
            if np_loss < min_loss:
                self.save(sess,"models/eejjj.ckpt")
                min_loss = np_loss
                print("Loss = %e" %(np_loss))
            if epoch % printout == 0:
                print("Epoch %4d: loss = %e, average integral = %e, average variance = %e, average stddev = %e"   
                        %(epoch, np_loss, np.mean(self.integrals), np.mean(self.vars),np.sqrt(np.mean(self.vars))))
                print("Epoch %4d: loss = %e, float.  integral = %e, float.  variance = %e, float.  stddev = %e"   
                        %(epoch, np_loss, np.mean(floating_int), np.mean(floating_av),np.sqrt(np.mean(floating_av))))
                figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
                plt.savefig('fig_{:04d}.pdf'.format(epoch))
                plt.close()
#            if np.sqrt(np_var)/np_integral < stopping:
#                break

        print("Epoch %4d: loss = %e, average integral = %e, average variance = %e, average stddev = %e"   
                %(epoch, np_loss, np.mean(self.integrals), np.mean(self.vars),np.sqrt(np.mean(self.vars))))
        print("Epoch %4d: loss = %e, float.  integral = %e, float.  variance = %e, float.  stddev = %e"   
                        %(epoch, np_loss, np.mean(floating_int), np.mean(floating_av),np.sqrt(np.mean(floating_av))))
        figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('fig_{:04d}.pdf'.format(epoch))
        plt.close()
        self.load(sess,"models/eejjj.ckpt")
        np_loss, np_integral, np_var, xpts, ppts, qpts = sess.run([self.loss, self.integral, self.var, self.x, self.p, self.q])
        print("Loaded configuration: loss = %e, integral = %e, variance = %e, stddev %e" %(np_loss,np_integral,np_var,np.sqrt(np_var)))
        figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('fig_final.pdf')
        plt.close()
        
        
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
        ran = tf.cast(tf.floor(tf.random.uniform((nsamples,),minval=0., maxval=self.nchannels,dtype=tf.dtypes.float32)),tf.dtypes.int32)
        #ran = tf.cast(tf.floor(tf.random.uniform((nsamples,),minval=1., maxval=2.,dtype=tf.dtypes.float32)),tf.dtypes.int32)
        ran_np=ran.eval()

        randic = {"channel": ran_np.tolist()}
        layerdic = {"L":randic}
        kwargs={"bijector_kwargs":layerdic}
        x = self.dist.sample(nsamples,**kwargs)
        p = self.func(x,randic)
        q = self.dist.prob(x,**kwargs)

        integral, var = tf.nn.moments(p/q,axes=[0])
        error = tf.sqrt(var/nsamples)

        np_int, np_error, xpts, qpts, ppts = sess.run([integral,error,x,q,p])
        #figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], weights = 1./qpts, show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        #plt.savefig('xsec_final_invq.pdf')
        #plt.close()

        figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('xsec_final_x.pdf')
        plt.close()
        
        figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], weights=1./qpts,show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('xsec_final_xdivq.pdf')
        plt.close()
        """
        figure = corner.corner(xpts, labels=[r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$',r'$x_5$'], weights=ppts/qpts,show_titles=True, title_kwargs={"fontsize": 12}, range=self.ndims*[[0,1]])
        plt.savefig('xsec_final_xPdivq.pdf')
        plt.close()
        """
        wgt = ppts/(qpts*np_int)
        print("Unweighting in Integrate: "+str(np.mean(wgt)/np.max(wgt)))
        plt.hist(wgt,bins=np.logspace(np.log10(np.min(wgt)),np.log10(np.max(wgt)),100),log=True)
        plt.xscale("log")
        plt.savefig('unweighting_eff.pdf')
        plt.close()

        f = open("weights.dat","w")
        f.write("Integral: {:0.2f} +/- {:0.2f}\n\
mean / max : {:0.2f} % \n\
for {:d} points\n".format(np_int,np_error,100*np.mean(wgt)/np.max(wgt),nsamples))
        for i in wgt:
            f.write(str(i)+"\n")
        f.close()
        return np_int, np_error
        

if __name__ == '__main__':
    import tensorflow as tf
    def normalChristina(x):
        return 0.8* tf.exp((-0.5*((x[:,0]-0.5)* (50 *(x[:,0]-0.5) -  15* (x[:,1]-0.5)) + (-15*(x[:,0]-0.5) + 5*(x[:,1]-0.5))* (x[:,1]-0.5)))) + x[:,2]

    integrator = Integrator(normalChristina, 3, mode='quadratic_blob')
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
