import numpy as np

from vector import *
from channels import SChannelDecay, Propagator

import logging
logger = logging.getLogger('eejjj')

class eetojjj:

    def __init__(self,alphas,ecms=91.2):
        self.ecms = ecms
        self.prop = Propagator()
        self.decay = SChannelDecay()
        self.channel = 1

    def Channel1(self,rans,pa,pb):
        s13 = self.prop.GeneratePoint(1,self.ecms**2,rans[:,0])
        p13, p2 = self.decay.GeneratePoint(pa+pb,s13,0.0,np.array([rans[:,1], rans[:,2]]).T)
        p1, p3 = self.decay.GeneratePoint(p13,0.0,0.0,np.array([rans[:,3], rans[:,4]]).T)

        return p1, p2, p3

    def Channel2(self,rans,pa,pb):
        s23 = self.prop.GeneratePoint(1,self.ecms**2,rans[:,0])
        p23, p1 = self.decay.GeneratePoint(pa+pb,s23,0.0,np.array([rans[:,1], rans[:,2]]).T)
        p2, p3 = self.decay.GeneratePoint(p23,0.0,0.0,np.array([rans[:,3], rans[:,4]]).T)

        return p1, p2, p3

    def ChannelIso(self,rans,pa,pb):
        s12 = self.prop.GeneratePoint(1e-1,self.ecms**2,rans[:,2])
        p1, p2 = self.decay.GeneratePoint(pa+pb,s12,0.0,np.array([rans[:,0], rans[:,1]]).T)

        return p1, p2

    def GeneratePoint(self,rans):
        pa = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     self.ecms/2*np.ones(np.shape(rans)[0]))
        pb = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     -self.ecms/2*np.ones(np.shape(rans)[0]))

#        channels = np.random.randint(0,2,np.shape(rans)[0])
#
#        ch1_pts = np.where(channels == 1, True, False)

        if self.channel == 0:
            p1, p2, p3 = self.Channel1(rans,pa,pb)
        else:
            p1, p2, p3 = self.Channel2(rans,pa,pb)
        #if channel == 1:
        #    p1, p2, p3 = self.Channel1(rans,pa,pb)
        #else:
        #    p1, p2, p3 = self.Channel2(rans,pa,pb)

        p13 = p1+p3
        ws13 = self.prop.GenerateWeight(1,self.ecms**2,p13)
        wp13_2 = self.decay.GenerateWeight(pa+pb,Mass2(p13),0.0,p13,p2)
        wp1_3 = self.decay.GenerateWeight(p1+p3,0.0,0.0,p1,p3)

        p23 = p2+p3
        ws23 = self.prop.GenerateWeight(1,self.ecms**2,p23)
        wp23_1 = self.decay.GenerateWeight(pa+pb,Mass2(p23),0.0,p23,p1)
        wp2_3 = self.decay.GenerateWeight(p23,0.0,0.0,p2,p3)

        wsum = wp13_2*ws13*wp1_3 + wp23_1*ws23*wp2_3

        lome = sherpa.ME2(np.array([pa,pb,p3,p1,p2]))
        dxs = 5.0*lome*wsum

        return dxs

if __name__ == '__main__':
    from qcd import AlphaS
    from comix import Comix
    from integrator import *

    import logging
    logging.basicConfig()

    alphas = AlphaS(91.1876,0.118)
    hardxs = eetojjj(alphas)
    sherpa = Comix([11,-11],[1,-1,21])

    def func(x):
#        return tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x],tf.float32))
        nbatchs = 10
        results = []
        batch_size = x.shape[0]//nbatchs
        for i in range(nbatchs):
            hardxs.channel = np.random.randint(0,2)
            results.append(tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x[i*batch_size:(i+1)*batch_size]],tf.float32)))
        return tf.concat(results,0)
#        hardxs.channel = 1
#        channel1 = tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x[:x.shape[0]//2]],tf.float32))
#        hardxs.channel = 2
#        channel2 = tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x[x.shape[0]//2:]],tf.float32))
#        return tf.concat([channel1,channel2],0)

    import corner

    x1 = np.random.random((100000,5))
    p1 = hardxs.GeneratePoint(x1)

    x2 = np.random.random((100000,5))
    hardxs.channel = 2
    p2 = hardxs.GeneratePoint(x2)

    x = np.concatenate([x1,x2])
    p = np.concatenate([p1,p2])
#    p = p/np.mean(p)
    figure = corner.corner(x, labels=[r'$x_0$',r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$'], weights=p, show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('matrix.pdf')

    import tensorflow as tf

    integrator = Integrator(func, 5, mode='linear')
    integrator.make_optimizer(nsamples=1000, learning_rate=1e-3)

    with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
        sess.run(tf.global_variables_initializer())
        integrator.optimize(sess,epochs=400,printout=10)
        print(integrator.integrate(sess,10000))

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,5))
    ax1 = integrator.plot_loss(ax1)
    ax2 = integrator.plot_integral(ax2)
    ax3 = integrator.plot_variance(ax3)
    plt.savefig('loss.pdf') 
