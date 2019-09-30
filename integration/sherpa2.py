import numpy as np

from flow.phase_space.vector import *
from flow.phase_space.channels import SChannelDecay, Propagator

import logging
logger = logging.getLogger('eejjj')

hbarc2 = 0.3893e9
mt = 173.21
gt = 2
mw = 80.385

class eetojjj:

    def __init__(self,alphas,ecms=91.2):
        self.cutoff = 1e-2
        self.ecms = ecms
        self.prop = Propagator(0.01)
        self.decayIso = SChannelDecay()
        self.decayGluon = SChannelDecay(0.99)
        self.channels = {}
#        self.channels[0] = self.ChannelIso
        self.channels[0] = self.ChannelA1
        self.channels[1] = self.ChannelA2
        self.channels[2] = self.ChannelA3
        self.channels[3] = self.ChannelA4
        self.channels[4] = self.ChannelC1
        self.channels[5] = self.ChannelC2
        self.channels[6] = self.ChannelC3
        self.channels[7] = self.ChannelC4
        self.channels[8] = self.ChannelB1
        self.channels[9] = self.ChannelB2
        self.channels[10] = self.ChannelB3
        self.channels[11] = self.ChannelB4

        self.weights = {}
        self.weights[0] = self.WeightA1
        self.weights[1] = self.WeightA2
        self.weights[2] = self.WeightA3
        self.weights[3] = self.WeightA4
        self.weights[4] = self.WeightC1
        self.weights[5] = self.WeightC2
        self.weights[6] = self.WeightC3
        self.weights[7] = self.WeightC4
        self.weights[8] = self.WeightB1
        self.weights[9] = self.WeightB2
        self.weights[10] = self.WeightB3
        self.weights[11] = self.WeightB4

        self.index = 0

    def PhaseSpace(self,channels,rans,pa,pb):
        logger.debug('channels = {}'.format(channels))
        s1min = self.cutoff
        s1max = tf.where(channels < 4, (self.ecms-self.cutoff)**2, self.ecms**2)
        s1 = self.prop.GeneratePoint(s1min,s1max,rans[:,0])

        s2min = self.cutoff
        s2max = tf.where(channels < 4, (self.ecms-tf.sqrt(s1))**2, s1)
        s2 = self.prop.GeneratePoint(s2min,s2max,rans[:,1])

        q1, q2 = self.decayIso.GeneratePoint(pa+pb, s1,
                tf.where(channels < 4, s2, 0),
                tf.transpose(tf.convert_to_tensor([rans[:,2], rans[:,3]])))

        q3, q4 = self.decayGluon.GeneratePoint(q1,
                tf.where(np.logical_or(channels < 4,channels > 7), 0, s2),
                tf.where(channels < 8, 0, s2),
                tf.transpose(tf.convert_to_tensor([rans[:,4], rans[:,5]])))

        q5, q6 = self.decayGluon.GeneratePoint(tf.where(channels[:,tf.newaxis] < 4, q2,
            np.where(channels[:,np.newaxis] < 8, q3, q4)),
            0.0, 0.0, np.array([rans[:,6],rans[:,7]]).T)

        condlist = [channels[:,np.newaxis] < 4, channels[:,np.newaxis] < 8, channels[:,np.newaxis] < 12]

        p1choice = [q3,q5,q3]
        p2choice = [q5,q2,q2]
        p3choice = [q4,q6,q5]
        p4choice = [q6,q4,q6]

        k1 = np.select(condlist, p1choice)
        k2 = np.select(condlist, p2choice)
        k3 = np.select(condlist, p3choice)
        k4 = np.select(condlist, p4choice)

        condlist_perm = [np.mod(channels[:,np.newaxis], 4) == 0, 
                np.mod(channels[:,np.newaxis], 4) == 1,
                np.mod(channels[:,np.newaxis], 4) == 2,
                np.mod(channels[:,np.newaxis], 4) == 3]

        p1perm = [k1,k1,k2,k2]
        p2perm = [k2,k2,k1,k1]
        p3perm = [k3,k4,k3,k4]
        p4perm = [k4,k3,k4,k3]

        p1 = np.select(condlist_perm, p1perm)
        p2 = np.select(condlist_perm, p2perm)
        p3 = np.select(condlist_perm, p3perm)
        p4 = np.select(condlist_perm, p4perm)

        return p1, p2, p3, p4

    def GenerateMomenta(self,channels,rans,pa,pb):
        return self.ChannelA1(rans,pa,pb)

    def ChannelA(self,rans,pa,pb):
        s134 = self.prop.GeneratePoint(self.cutoff,self.ecms**2,rans[:,0])
        s13 = self.prop.GeneratePoint(self.cutoff,s134,rans[:,1])
        p134, p2 = self.decayIso.GeneratePoint(pa+pb,s134,0.0,np.array([rans[:,2], rans[:,3]]).T)
        p13, p4 = self.decayGluon.GeneratePoint(p134,s13,0.0,np.array([rans[:,4], rans[:,5]]).T)
        p1, p3 = self.decayGluon.GeneratePoint(p13,0.0,0.0,np.array([rans[:,6], rans[:,7]]).T)

        return p1, p2, p3, p4

    def ChannelB(self,rans,pa,pb):
        s13 = self.prop.GeneratePoint(self.cutoff,(self.ecms-self.cutoff)**2,rans[:,0])
        s24 = self.prop.GeneratePoint(self.cutoff,(self.ecms-np.sqrt(s13))**2,rans[:,1])
        p13, p24 = self.decayIso.GeneratePoint(pa+pb,s13,s24,np.array([rans[:,2], rans[:,3]]).T)
        p1, p3 = self.decayGluon.GeneratePoint(p13,0.0,0.0,np.array([rans[:,4], rans[:,5]]).T)
        p2, p4 = self.decayGluon.GeneratePoint(p24,0.0,0.0,np.array([rans[:,6], rans[:,7]]).T)

        return p1, p2, p3, p4

    def ChannelC(self,rans,pa,pb):
        s134 = self.prop.GeneratePoint(self.cutoff,self.ecms**2,rans[:,0])
        s34 = self.prop.GeneratePoint(self.cutoff,s134,rans[:,1])
        p134, p2 = self.decayIso.GeneratePoint(pa+pb,s134,0.0,np.array([rans[:,2], rans[:,3]]).T)
        p1, p34 = self.decayGluon.GeneratePoint(p134,0.0,s34,np.array([rans[:,4], rans[:,5]]).T)
        p3, p4 = self.decayGluon.GeneratePoint(p34,0.0,0.0,np.array([rans[:,6], rans[:,7]]).T)

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
        p2, p1, p4, p3 = self.ChannelA(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelC1(self,rans,pa,pb):
        p1, p2, p3, p4 = self.ChannelC(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelC2(self,rans,pa,pb):
        p1, p2, p4, p3 = self.ChannelC(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelC3(self,rans,pa,pb):
        p2, p1, p3, p4 = self.ChannelC(rans,pa,pb)
        return p1, p2, p3, p4

    def ChannelC4(self,rans,pa,pb):
        p2, p1, p4, p3 = self.ChannelC(rans,pa,pb)
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
        ws134 = self.prop.GenerateWeight(self.cutoff,self.ecms**2,p134)
        ws13 = self.prop.GenerateWeight(self.cutoff,Dot(p134,p134),p13)
        wp134_2 = self.decayIso.GenerateWeight(pa+pb,Mass2(p134),0.0,p134,p2)
        wp13_4 = self.decayGluon.GenerateWeight(p13+p4,Mass2(p13),0.0,p13,p4)
        wp1_3 = self.decayGluon.GenerateWeight(p1+p3,0.0,0.0,p1,p3)

        return np.maximum(ws134*ws13*wp134_2*wp13_4*wp1_3,1e-7)

    def Weight2(self,pa,pb,p1,p2,p3,p4):
        p13 = p1+p3
        p24 = p2+p4
        ws13 = self.prop.GenerateWeight(self.cutoff,self.ecms**2,p13)
        ws24 = self.prop.GenerateWeight(self.cutoff,(self.ecms-Mass(p13))**2,p24)
        wp13_24 = self.decayIso.GenerateWeight(pa+pb,Mass2(p13),Mass2(p24),p13,p24)
        wp1_3 = self.decayGluon.GenerateWeight(p1+p3,0.0,0.0,p1,p3)
        wp2_4 = self.decayGluon.GenerateWeight(p2+p4,0.0,0.0,p2,p4)

        return np.maximum(ws24*ws13*wp13_24*wp2_4*wp1_3,1e-7)

    def Weight3(self,pa,pb,p1,p2,p3,p4):
        p34 = p3+p4
        p134 = p34+p1
        ws134 = self.prop.GenerateWeight(self.cutoff,self.ecms**2,p134)
        ws34 = self.prop.GenerateWeight(self.cutoff,Dot(p134,p134),p34)
        wp134_2 = self.decayIso.GenerateWeight(pa+pb,Mass2(p134),0.0,p134,p2)
        wp1_34 = self.decayGluon.GenerateWeight(p34+p1,0.0,Mass2(p34),p1,p34)
        wp3_4 = self.decayGluon.GenerateWeight(p3+p4,0.0,0.0,p3,p4)

        return np.maximum(ws134*ws34*wp134_2*wp1_34*wp3_4,1e-7)

    def WeightA1(self,pa,pb,p1,p2,p3,p4):
        return self.Weight1(pa,pb,p1,p2,p3,p4)

    def WeightA2(self,pa,pb,p1,p2,p3,p4):
        return self.Weight1(pa,pb,p1,p2,p4,p3)

    def WeightA3(self,pa,pb,p1,p2,p3,p4):
        return self.Weight1(pa,pb,p2,p1,p3,p4)

    def WeightA4(self,pa,pb,p1,p2,p3,p4):
        return self.Weight1(pa,pb,p2,p1,p4,p3)

    def WeightB1(self,pa,pb,p1,p2,p3,p4):
        return self.Weight2(pa,pb,p1,p2,p3,p4)

    def WeightB2(self,pa,pb,p1,p2,p3,p4):
        return self.Weight2(pa,pb,p1,p2,p4,p3)

    def WeightB3(self,pa,pb,p1,p2,p3,p4):
        return self.Weight2(pa,pb,p2,p1,p3,p4)

    def WeightB4(self,pa,pb,p1,p2,p3,p4):
        return self.Weight2(pa,pb,p2,p1,p4,p3)

    def WeightC1(self,pa,pb,p1,p2,p3,p4):
        return self.Weight3(pa,pb,p1,p2,p3,p4)

    def WeightC2(self,pa,pb,p1,p2,p3,p4):
        return self.Weight3(pa,pb,p1,p2,p4,p3)

    def WeightC3(self,pa,pb,p1,p2,p3,p4):
        return self.Weight3(pa,pb,p2,p1,p3,p4)

    def WeightC4(self,pa,pb,p1,p2,p3,p4):
        return self.Weight3(pa,pb,p2,p1,p4,p3)

    def ChannelIso(self,rans,pa,pb):
        p1, p2 = self.decayIso.GeneratePoint(pa+pb,0,0,rans)

        return p1, p2

    def WeightIso(self,pa,pb,p1,p2):
        return self.decayIso.GenerateWeight(pa+pb,0,0,p1,p2)

    def GeneratePointIso(self,rans):
        pa = Vector4(self.ecms/2*tf.ones(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                self.ecms/2*tf.ones(tf.shape(rans)[0]))
        pb = Vector4(self.ecms/2*tf.ones(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                -self.ecms/2*tf.ones(tf.shape(rans)[0]))

        p1, p2 = self.ChannelIso(rans,pa,pb)

        wsum = self.WeightIso(pa,pb,p1,p2)
        momentum = tf.stack([pa,pb,p1,p2])
        #lome = tf.convert_to_tensor(sherpa.process.CSMatrixElementVec(momentum), dtype=tf.float64)
        lome = self.CallSherpa(momentum)
        print(lome)
        tf.print(lome)
        dxs = lome*wsum*hbarc2/self.ecms**2/2.0

        return dxs

    def CallSherpa(self, momentum):
        return tf.py_function(sherpa.process.CSMatrixElementVec2, [momentum, 1000], tf.float64)


    def ChannelTT(self, rans, pa, pb):
        s13 = self.prop.GeneratePoint((mw+self.cutoff)**2,(self.ecms-mw-self.cutoff)**2,rans[:,0],mass=mt,width=gt)
        s24 = self.prop.GeneratePoint((mw+self.cutoff)**2,(self.ecms-tf.sqrt(s13))**2,rans[:,1],mass=mt,width=gt)

        p13, p24 = self.decayIso.GeneratePoint(pa+pb,s13,s24,rans[:,2:4]) 
        p1, p3 = self.decayIso.GeneratePoint(p13,mw**2,0.0,rans[:,4:6])
        p2, p4 = self.decayIso.GeneratePoint(p24,mw**2,0.0,rans[:,6:])

        return p1, p2, p3, p4

    def WeightTT(self, pa, pb, p1, p2, p3, p4):
        p13 = p1+p3
        p24 = p2+p4
        ws13 = self.prop.GenerateWeight((mw+self.cutoff)**2,(self.ecms-mw-self.cutoff)**2,p13,mass=mt,width=gt)
        ws24 = self.prop.GenerateWeight((mw+self.cutoff)**2,(self.ecms-Mass(p13))**2,p24,mass=mt,width=gt)
        wp13_24 = self.decayIso.GenerateWeight(pa+pb,Mass2(p13),Mass2(p24),p13,p24)
        wp1_3 = self.decayIso.GenerateWeight(p1+p3,mw**2,0.0,p1,p3)
        wp2_4 = self.decayIso.GenerateWeight(p2+p4,mw**2,0.0,p2,p4)

        return tf.maximum(ws24*ws13*wp13_24*wp2_4*wp1_3,1e-7)
        
    def GeneratePointTT(self, rans):
        pa = Vector4(self.ecms/2*tf.ones(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                self.ecms/2*tf.ones(tf.shape(rans)[0]))
        pb = Vector4(self.ecms/2*tf.ones(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                tf.zeros(tf.shape(rans)[0]),
                -self.ecms/2*tf.ones(tf.shape(rans)[0]))

        p1,p2,p3,p4 = self.ChannelTT(rans,pa,pb)

        logger.debug('p1 = {}, m1 = {}'.format(p1,Mass2(p1)))
        logger.debug('p2 = {}, m2 = {}'.format(p2,Mass2(p2)))
        logger.debug('p3 = {}, m3 = {}'.format(p3,Mass2(p3)))
        logger.debug('p4 = {}, m4 = {}'.format(p4,Mass2(p4)))

        wsum = self.WeightTT(pa, pb, p1, p2, p3, p4)

        lome = tf.convert_to_tensor(self.CallSherpa(momentum), dtype=tf.float64)

        logger.debug('wsum = {}'.format(wsum))
        logger.debug('lome = {}'.format(lome))
        logger.debug('ecm = {}'.format(self.ecms))

        dxs = lome*wsum*hbarc2/self.ecms**2/2.0
        """
        plt.hist(lome,bins=np.logspace(-10,5,500))
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('figs/ME_{:04d}.pdf'.format(self.index))
        plt.close()

        plt.hist(dxs,bins=np.logspace(-6,10,500))
        plt.yscale('log')
        plt.xscale('log')
        plt.savefig('figs/dsig_{:04d}.pdf'.format(self.index))
        plt.close()
        """
        self.index += 1

        #print(np.mean(dxs))

        return tf.maximum(dxs,1e-7)

    def GeneratePoint(self,rans,channel):
        pa = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                self.ecms/2*np.ones(np.shape(rans)[0]))
        pb = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                -self.ecms/2*np.ones(np.shape(rans)[0]))

        p1, p2, p3, p4 = self.PhaseSpace(channel,rans,pa,pb)
        logger.debug(p1)

        wsum = np.array([self.Weight1(pa,pb,p1,p2,p3,p4), self.Weight1(pa,pb,p1,p2,p4,p3),
            self.Weight1(pa,pb,p2,p1,p3,p4), self.Weight1(pa,pb,p2,p1,p4,p3),
            self.Weight2(pa,pb,p1,p2,p3,p4), self.Weight2(pa,pb,p1,p2,p4,p3),
            self.Weight2(pa,pb,p2,p1,p3,p4), self.Weight2(pa,pb,p2,p1,p4,p3),
            self.Weight3(pa,pb,p1,p2,p3,p4), self.Weight3(pa,pb,p1,p2,p4,p3),
            self.Weight3(pa,pb,p2,p1,p3,p4), self.Weight3(pa,pb,p2,p1,p4,p3)])


        wsum = np.reciprocal(np.mean(np.reciprocal(wsum),axis=0))

        lome = np.array(sherpa.process.CSMatrixElementVec(np.array([pa,pb,p3,p4,p1,p2])))
        dxs = lome*wsum*hbarc2/self.ecms**2/2.0

        return np.maximum(np.nan_to_num(dxs),1e-7)

if __name__ == '__main__':
    from qcd import AlphaS
    from comix import Comix
    from flow.integration import integrator
    from flow.integration import couplings
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    tfd = tfp.distributions
    import matplotlib.pyplot as plt
    import os
    import argparse
    import time

    import corner

    import logging

    parser = argparse.ArgumentParser(
            description='Sherpa Event Generator with Flow integration.'
            )

    parser.add_argument('-d','--debug',action='store_true',
            help='Turn on debug logging')
    parser.add_argument('-e','--epochs',default=500,
            help='Number of epochs to train for')
    parser.add_argument('-p','--plot',action='store_true',
            help='Flag for plotting distributions')
    parser.add_argument('-l','--loss',action='store_true',
            help='Flag for plotting loss, integral, and variance')
    parser.add_argument('-n','--nsamples',default=2000,
            help='Number of samples to calculate the loss')
    parser.add_argument('-a','--acceptance',action='store_true',
            help='Calculate acceptance')

    options = parser.parse_args()
    acceptance = options.acceptance
    plot = options.plot

    logging.basicConfig()
    if options.debug:
        logging.getLogger('eejjj').setLevel(logging.DEBUG)
#        logging.getLogger('channels').setLevel(logging.DEBUG)

#    tf.config.experimental_run_functions_eagerly(True)

    alphas = AlphaS(91.1876,0.118)
    hardxs = eetojjj(alphas,91.18)
    in_part = [11,-11]
#    out_part = [1,-1,21,21]
#    out_part = [5,-5,24,-24]
    out_part = [1,-1]
    npart = len(out_part)
    ndims = 3*npart - 4
    sherpa = Comix(in_part,out_part)

    start = time.perf_counter()
    nevents = 10000
    x = np.random.random((nevents,ndims))
    p = hardxs.GeneratePointIso(x)

    first = time.perf_counter()

    nevents = 10000
    x = np.random.random((nevents,ndims))
    p = hardxs.GeneratePointIso(x)

    second = time.perf_counter()

    time1 = first-start
    time2 = second-first

    print(time1, time2, time1/time2)

    figure = corner.corner(x, labels=[r'$x_{}$'.format(i) for i in range(ndims)], weights=p, show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig('matrix.png')
    plt.close()

    nint = 10000

    def build_dense(in_features, out_features):
        invals = tf.keras.layers.Input(in_features, dtype=tf.float64)
#        h_widths = tf.keras.layers.Dense(128)(invals)
#        h_heights = tf.keras.layers.Dense(128)(invals)
#        h_derivs = tf.keras.layers.Dense(128)(invals)
#        h_widths = tf.keras.layers.Dense((out_features-1)/3)(h_widths)
#        h_heights = tf.keras.layers.Dense((out_features-1)/3)(h_heights)
#        h_derivs = tf.keras.layers.Dense((out_features-1)/3+2)(h_derivs)
#        outputs = tf.keras.layers.Concatenate()([h_widths, h_heights, h_derivs])
        h = tf.keras.layers.Dense(128)(invals)
        h = tf.keras.layers.Dense(128)(h)
        h = tf.keras.layers.Dense(128)(h)
        h = tf.keras.layers.Dense(128)(h)
        h = tf.keras.layers.Dense(128)(h)
        h = tf.keras.layers.Dense(128)(h)
        outputs = tf.keras.layers.Dense(out_features)(h)
        model = tf.keras.models.Model(invals,outputs)
        model.summary()
        return model

    ndims = 2

    bijectors = []
    masks = [[x % 2 for x in range(1,ndims+1)],[x % 2 for x in range(0,ndims)],[1 if x < ndims/2 else 0 for x in range(0,ndims)],[0 if x < ndims/2 else 1 for x in range(0,ndims)]]
    bijectors.append(couplings.PiecewiseRationalQuadratic([1,0],build_dense,num_bins=100,blob=32))
    bijectors.append(couplings.PiecewiseRationalQuadratic([0,1],build_dense,num_bins=100,blob=32))
    
    bijectors = tfb.Chain(list(reversed(bijectors)))
    
    base_dist = tfd.Uniform(low=ndims*[tf.constant(0.,dtype=tf.float64)], high=ndims*[1.])
    base_dist = tfd.Independent(distribution=base_dist,
                                reinterpreted_batch_ndims=1,
                                )
    dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=bijectors,
    )

    initial_learning_rate = 1e-4
    
    optimizer = tf.keras.optimizers.Adam(initial_learning_rate, clipnorm = 5.0)#lr_schedule)
    
    integrator = integrator.Integrator(hardxs.GeneratePointIso, dist, optimizer)



#    print(integrator.integrate(nint,
#          acceptance=acceptance,
#          fname='untrained',
#          plot=True,
#          min=1e-9,
#          max=1e3,
#          nbins=300))

    nsamples = []
    nsamples.extend([1000]*200)
    nsamples.extend([2000]*200)
    nsamples.extend([4000]*200)
    nsamples.extend([8000]*200)
#    nsamples.extend([8000]*100)
#    nsamples.extend([16000]*500)
    losses = []
    integrals = []
    errors = []
    min_loss = 1e99
    nsamples = 1000
    epochs = 100

    try:
        for epoch in range(epochs):
            if epoch % 5 == 0:
                samples = integrator.sample(10000)
                hist2d_kwargs={'smooth':2}
                figure = corner.corner(samples, labels=[r'$x_1$',r'$x_2$'], show_titles=True, title_kwargs={"fontsize": 12}, range=ndims*[[0,1]],**hist2d_kwargs)

            loss, integral, error = integrator.train_one_step(nsamples,integral=True)
            if epoch % 5 == 0:
                figure.suptitle('loss = '+str(loss.numpy()),fontsize=16,x = 0.75)
                plt.savefig('fig_{:04d}.png'.format(epoch))
                plt.close()
            losses.append(loss)
            integrals.append(integral)
            errors.append(error)
            if loss < min_loss:
                min_loss = loss
                integrator.save()
            if epoch % 10 == 0:
                print(epoch, loss.numpy(), integral.numpy(), error.numpy())
    except KeyboardInterrupt:
        pass
        
    integrator.load()
    
    weights = []
    for i in range(10):
        weights.append(integrator.acceptance(100000).numpy())
    weights = np.concatenate(weights)

    # Remove outliers
    #weights = np.sort(weights)
    #weights = np.where(weights < np.mean(weights)*0.01, 0, weights)

    average = np.mean(weights)
    max_wgt = np.max(weights)

    print("acceptance = "+str(average/max_wgt))

    plt.hist(weights,bins=np.logspace(np.log10(np.minimum(weights)),np.log10(np.maximum(weights)),100))
    plt.axvline(average,linestyle='--',color='red')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig('efficiency.png')
    plt.show()

    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('loss.png')
    plt.show()
