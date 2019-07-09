import numpy as np

from vector import *
from channels import SChannelDecay, Propagator

import logging
logger = logging.getLogger('eejjj')

hbarc2 = 0.3893e9

class eetojjj:

    def __init__(self,alphas,ecms=91.2):
        self.ecms = ecms
        self.prop = Propagator()
        self.decay = SChannelDecay()
        self.channels = {}
#        self.channels[0] = self.ChannelIso
#        self.channels[1] = self.ChannelIso
#        self.channels[2] = self.ChannelIso
#        self.channels[3] = self.ChannelIso
#        self.channels[4] = self.ChannelIso
#        self.channels[5] = self.ChannelIso
#        self.channels[6] = self.ChannelIso
#        self.channels[7] = self.ChannelIso
        self.channels[0] = self.ChannelA1
        self.channels[1] = self.ChannelA2
        self.channels[2] = self.ChannelA3
        self.channels[3] = self.ChannelA4
        self.channels[4] = self.ChannelA5
        self.channels[5] = self.ChannelA6
        self.channels[6] = self.ChannelA7
        self.channels[7] = self.ChannelA8
        self.channels[8] = self.ChannelB1
        self.channels[9] = self.ChannelB2
        self.channels[10] = self.ChannelB3
        self.channels[11] = self.ChannelB4

    def GenerateMomenta(self,channel,rans,pa,pb):
        return self.channels[channel](rans,pa,pb)

    def ChannelA(self,rans,pa,pb):
        s134 = self.prop.GeneratePoint(1,self.ecms**2,rans[:,0])
        s13 = self.prop.GeneratePoint(1,s134,rans[:,1])
        p134, p2 = self.decay.GeneratePoint(pa+pb,s134,0.0,np.array([rans[:,2], rans[:,3]]).T)
        p13, p4 = self.decay.GeneratePoint(p134,s13,0.0,np.array([rans[:,4], rans[:,5]]).T)
        p1, p3 = self.decay.GeneratePoint(p13,0.0,0.0,np.array([rans[:,6], rans[:,7]]).T)

        return p1, p2, p3, p4

    def ChannelB(self,rans,pa,pb):
        s13 = self.prop.GeneratePoint(1,(self.ecms-1)**2,rans[:,0])
        s24 = self.prop.GeneratePoint(1,(self.ecms-np.sqrt(s13))**2,rans[:,1])
        p13, p24 = self.decay.GeneratePoint(pa+pb,s13,s24,np.array([rans[:,2], rans[:,3]]).T)
        p1, p3 = self.decay.GeneratePoint(p13,0.0,0.0,np.array([rans[:,4], rans[:,5]]).T)
        p2, p4 = self.decay.GeneratePoint(p24,0.0,0.0,np.array([rans[:,6], rans[:,7]]).T)

        return p1, p2, p3, p4

    def ChannelA1(self,rans,pa,pb):
        p1, p2, p3, p4 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA2(self,rans,pa,pb):
        p1, p2, p4, p3 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA3(self,rans,pa,pb):
        p2, p1, p3, p4 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA4(self,rans,pa,pb):
        p2, p1, p3, p4 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA5(self,rans,pa,pb):
        p3, p2, p1, p4 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA6(self,rans,pa,pb):
        p3, p1, p2, p4 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA7(self,rans,pa,pb):
        p4, p2, p1, p3 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelA8(self,rans,pa,pb):
        p4, p1, p2, p3 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelB1(self,rans,pa,pb):
        p1, p2, p3, p4 = self.ChannelB(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelB2(self,rans,pa,pb):
        p2, p1, p3, p4 = self.ChannelB(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelB3(self,rans,pa,pb):
        p1, p2, p4, p3 = self.ChannelB(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelB4(self,rans,pa,pb):
        p2, p1, p4, p3 = self.ChannelB(rans,pa,pb)
        return p1, p2, p3, p4

    def Weight1(self,pa,pb,p1,p2,p3,p4):
        p13 = p1+p3
        p134 = p13+p4
        ws134 = self.prop.GenerateWeight(1,self.ecms**2,p134)
        ws13 = self.prop.GenerateWeight(1,Dot(p134,p134),p13)
        wp134_2 = self.decay.GenerateWeight(pa+pb,Mass2(p134),0.0,p134,p2)
        wp13_4 = self.decay.GenerateWeight(p13+p4,Mass2(p13),0.0,p13,p4)
        wp1_3 = self.decay.GenerateWeight(p1+p3,0.0,0.0,p1,p3)

        return ws134*ws13*wp134_2*wp13_4*wp1_3

    def Weight2(self,pa,pb,p1,p2,p3,p4):
        p13 = p1+p3
        p24 = p2+p4
        ws13 = self.prop.GenerateWeight(1,self.ecms**2,p13)
        ws24 = self.prop.GenerateWeight(1,self.ecms**2-Mass2(p13),p24)
        wp13_24 = self.decay.GenerateWeight(pa+pb,Mass2(p13),Mass2(p24),p13,p24)
        wp1_3 = self.decay.GenerateWeight(p1+p3,0.0,0.0,p1,p3)
        wp2_4 = self.decay.GenerateWeight(p2+p4,0.0,0.0,p2,p4)

        return ws24*ws13*wp13_24*wp2_4*wp1_3

    def ChannelIso(self,rans,pa,pb):
#        s12 = self.prop.GeneratePoint(1e-1,self.ecms**2,rans[:,2])
        p1, p2 = self.decay.GeneratePoint(pa+pb,0.0,0.0,np.array([rans[:,0], rans[:,1]]).T)

        return p1, p2

    def WeightIso(self,pa,pb,p1,p2):
        return self.decay.GenerateWeight(pa+pb,0.0,0.0,p1,p2)

    def GeneratePoint2(self,rans):
        pa = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     self.ecms/2*np.ones(np.shape(rans)[0]))
        pb = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     -self.ecms/2*np.ones(np.shape(rans)[0]))
    
        channels = np.random.randint(0,len(self.channels),(rans.shape[0]))
        unique, counts = np.unique(channels, return_counts = True)
        counts = np.cumsum(counts)
        channels = dict(zip(unique, counts))

#        p1, p2 = self.ChannelIso(rans,pa,pb)
        p1 = []
        p2 = []
        position_old = 0
        for channel, position in channels.items():
            p1tmp, p2tmp = self.GenerateMomenta(channel,rans[position_old:position],
                                                pa[position_old:position],
                                                pb[position_old:position])
            p1.append(p1tmp)
            p2.append(p2tmp)
            position_old = position

        p1 = np.concatenate(p1)
        p2 = np.concatenate(p2)

        wsum = self.WeightIso(pa,pb,p1,p2)
        lome = np.array(sherpa.process.MatrixElementVec(np.array([pa,pb,p1,p2])))
        dxs = lome*wsum*hbarc2/self.ecms**2/2.0

        return np.nan_to_num(dxs)

    def GeneratePoint(self,rans):
        pa = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     self.ecms/2*np.ones(np.shape(rans)[0]))
        pb = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     -self.ecms/2*np.ones(np.shape(rans)[0]))

        channels = np.random.randint(0,len(self.channels),(rans.shape[0]))
        unique, counts = np.unique(channels, return_counts = True)
        counts = np.cumsum(counts)
        channels = dict(zip(unique, counts))

#        p1, p2 = self.ChannelIso(rans,pa,pb)
        p1 = []
        p2 = []
        p3 = []
        p4 = []
        position_old = 0
        for channel, position in channels.items():
            p1tmp, p2tmp, p3tmp, p4tmp = self.GenerateMomenta(
                                                channel,rans[position_old:position],
                                                pa[position_old:position],
                                                pb[position_old:position])
            p1.append(p1tmp)
            p2.append(p2tmp)
            p3.append(p3tmp)
            p4.append(p4tmp)
            position_old = position

        p1 = np.concatenate(p1)
        p2 = np.concatenate(p2)
        p3 = np.concatenate(p3)
        p4 = np.concatenate(p4)

        wsum = np.array([self.Weight1(pa,pb,p1,p2,p3,p4), self.Weight1(pa,pb,p1,p2,p4,p3),
                         self.Weight1(pa,pb,p2,p1,p3,p4), self.Weight1(pa,pb,p2,p1,p4,p3),
                         self.Weight2(pa,pb,p1,p2,p3,p4), self.Weight2(pa,pb,p1,p2,p4,p3),
                         self.Weight2(pa,pb,p2,p1,p3,p4), self.Weight2(pa,pb,p2,p1,p4,p3),
                         self.Weight1(pa,pb,p3,p2,p1,p4), self.Weight1(pa,pb,p3,p1,p2,p4),
                         self.Weight1(pa,pb,p4,p2,p1,p3), self.Weight1(pa,pb,p4,p1,p2,p3)])

        wsum = np.reciprocal(np.mean(np.reciprocal(wsum),axis=0))

        lome = np.array(sherpa.process.MatrixElementVec(np.array([pa,pb,p3,p4,p1,p2])))
        dxs = lome*wsum*hbarc2/self.ecms**2/2.0

        return np.maximum(np.nan_to_num(dxs),1e-7)

if __name__ == '__main__':
    from qcd import AlphaS
    from comix import Comix
    from integrator import *
    import tensorflow as tf

    import logging
    logging.basicConfig()
    logging.getLogger('eejjj').setLevel(logging.DEBUG)
#    logging.getLogger('channels').setLevel(logging.DEBUG)
    

    alphas = AlphaS(91.1876,0.118)
    hardxs = eetojjj(alphas)
    in_part = [11,-11]
    out_part = [1,-1,21,21]
#    out_part = [1,-1]
    npart = len(out_part)
    ndims = 3*npart - 4
    sherpa = Comix(in_part,out_part)

#    x = np.random.random((100000,ndims))
#    p = hardxs.GeneratePoint2(x)
#
#    figure = corner.corner(x, labels=[r'$x_{}$'.format(i) for i in range(ndims)], weights=p, show_titles=True, title_kwargs={"fontsize": 12})
#    plt.savefig('matrix.pdf')

    def func2(x):
        return tf.stop_gradient(tf.py_function(hardxs.GeneratePoint2,[x],tf.float32))

#    integrator = Integrator(func2, ndims, mode='linear')
#    integrator.make_optimizer(nsamples=1000, learning_rate=1e-3)
#
#    with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
#        try:
#            integrator.load(sess,"models/eejj.ckpt")
#        except: 
#            sess.run(tf.global_variables_initializer())
##            profiler = tf.profiler.Profiler(sess.graph)
##            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#            profiler = None
#            options = None
#            integrator.optimize(sess,epochs=20,printout=10,profiler=profiler,options=options)
#            integrator.save(sess,"models/eejj.ckpt")
#
#            if profiler is not None:
#                option_builder = tf.profiler.ProfileOptionBuilder
#                opts = (option_builder(option_builder.time_and_memory()).
#                        with_step(-1). # with -1, should compute the average of all registered steps.
#                        with_file_output('test.txt').
#                        select(['micros','bytes','occurrence']).order_by('micros').
#                        build())
#                # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
#                profiler.profile_operations(options=opts)
#        print(integrator.integrate(sess,100000))
#        print(integrator.acceptance(sess,100000))
#
#    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,5))
#    ax1 = integrator.plot_loss(ax1)
#    ax2 = integrator.plot_integral(ax2)
#    ax3 = integrator.plot_variance(ax3)
#    plt.savefig('loss.pdf') 
#
#    raise

    def func(x):
        return tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x],tf.float32))

    x = np.random.random((100000,ndims))
    p = hardxs.GeneratePoint(x)

    figure = corner.corner(x, labels=[r'$x_{}$'.format(i) for i in range(ndims)], weights=p, show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('matrix.pdf')

    import tensorflow as tf

    integrator = Integrator(func, ndims, mode='linear')
    integrator.make_optimizer(nsamples=2000, learning_rate=1e-3)

    with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
        sess.run(tf.global_variables_initializer())
        profiler = None
        options = None
#        profiler = tf.profiler.Profiler(sess.graph)
#        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        integrator.optimize(sess,epochs=1000,printout=10,profiler=profiler,options=options)
        print(integrator.integrate(sess,10000))

        if profiler is not None:
            option_builder = tf.profiler.ProfileOptionBuilder
            opts = (option_builder(option_builder.time_and_memory()).
                    with_step(-1). # with -1, should compute the average of all registered steps.
                    with_file_output('test.txt').
                    select(['micros','bytes','occurrence']).order_by('micros').
                    build())
            # Profiling infos about ops are saved in 'test-%s.txt' % FLAGS.out
            profiler.profile_operations(options=opts)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,5))
    ax1 = integrator.plot_loss(ax1)
    ax2 = integrator.plot_integral(ax2)
    ax3 = integrator.plot_variance(ax3)
    plt.savefig('loss.pdf') 
