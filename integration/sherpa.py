
import numpy as np

from vector import *
from particle import Particle
from channels import SChannelDecay, Propagator
from durham import Algorithm

import logging
logger = logging.getLogger('eejjj')

class eetojjj:

    def __init__(self,alphas,ecms=91.2):
        self.ecms = ecms
        self.duralg = Algorithm()
        self.prop = Propagator()
        self.decay = SChannelDecay()

    def Channel1(self,rans,pa,pb):
        s13 = self.prop.GeneratePoint(1.e-1,self.ecms**2,rans[:,0])
        p13_2 = self.decay.GeneratePoint(pa+pb,s13,0.,np.array([rans[:,1], rans[:,2]]).T)
        p1_3 = self.decay.GeneratePoint(p13_2[0],0.,0.,np.array([rans[:,3], rans[:,4]]).T)
        p = np.array([ pa, pb, p1_3[0], p13_2[1], p1_3[1] ])

        return p

    def Channel2(self,rans,pa,pb):
        s23 = self.prop.GeneratePoint(1.e-1,self.ecms**2,rans[:,0])
        p23_1 = self.decay.GeneratePoint(pa+pb,s23,0.,np.array([ rans[:,1], rans[:,2] ]).T)
        p2_3 = self.decay.GeneratePoint(p23_1[0],0.,0.,np.array([ rans[:,3], rans[:,4] ]).T)
        p = np.array([ pa, pb, p23_1[1], p2_3[0], p2_3[1] ])

        return p

    def GeneratePoint(self,rans):
        pa = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                self.ecms/2*np.ones(np.shape(rans)[0]))
        pb = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                np.zeros(np.shape(rans)[0]),
                -self.ecms/2*np.ones(np.shape(rans)[0]))
        # The phase space generation
        p = self.Channel1(rans,pa,pb)
#        p = np.where(np.random.random((1,rans.shape[0],1)) < 0.5,
#                     self.Channel1(rans,pa,pb),
#                     self.Channel2(rans,pa,pb))
#        if np.random.random()<0.5:
##            logger.debug("(13)2")
#            s13 = self.prop.GeneratePoint(1.e-3,self.ecms**2,rans[:,0])
#            p13_2 = self.decay.GeneratePoint(pa+pb,s13,0.,np.array([rans[:,1], rans[:,2]]).T)
#            p1_3 = self.decay.GeneratePoint(p13_2[0],0.,0.,np.array([rans[:,3], rans[:,4]]).T)
#            p = np.array([ -pa, -pb, p1_3[0], p13_2[1], p1_3[1] ])
#        else:
##            logger.debug("1(23)")
#            s23 = self.prop.GeneratePoint(1.e-3,self.ecms**2,rans[:,0])
#            p23_1 = self.decay.GeneratePoint(pa+pb,s23,0.,np.array([ rans[:,1], rans[:,2] ]).T)
#            p2_3 = self.decay.GeneratePoint(p23_1[0],0.,0.,np.array([ rans[:,3], rans[:,4] ]).T)
#            p = np.array([ pa, pb, p23_1[1], p2_3[0], p2_3[1] ])
#        logger.debug("p_0 = {0}, m2 = {1}".format(p[2],Mass2(p[:,0])))
#        logger.debug("p_1 = {0}, m2 = {1}".format(p[3],Mass2(p[:,1])))
#        logger.debug("p_2 = {0}, m2 = {1}".format(p[4],Mass2(p[:,2])))
#        logger.debug("sum: {0}".format((-p[0]-p[1]+p[2]+p[3]+p[4])))
        # The phase space weight
        p13 = p[2]+p[4]
        ws13 = self.prop.GenerateWeight(1.e-1,self.ecms**2,p13)
        wp13_2 = self.decay.GenerateWeight(p[0]+p[1],Mass2(p13),0.,p[2]+p[4],p[3])
        wp1_2 = self.decay.GenerateWeight(p[2]+p[4],0.,0.,p[2],p[4])
#        logger.debug("weights {0} {1} {2}".format(ws13,wp13_2,wp1_2))
        p23 = p[3]+p[4]
        ws23 = self.prop.GenerateWeight(1.e-1,self.ecms**2,p23)
        wp23_1 = self.decay.GenerateWeight(p[0]+p[1],Mass2(p23),0.,p[3]+p[4],p[2])
        wp2_3 = self.decay.GenerateWeight(p[3]+p[4],0.,0.,p[3],p[4])
#        logger.debug("weights {0} {1} {2}".format(ws23,wp23_1,wp2_3))
        wsum = wp13_2[0]*ws13[0]*wp1_2[0]#+wp23_1[0]*ws23[0]*wp2_3[0]
#        logger.debug("total weight {0}".format(wsum))
        rans1 = [ws13[1],wp13_2[1][0],wp13_2[1][1],wp1_2[1][0],wp1_2[1][1]]
        rans2 = [ws23[1],wp23_1[1][0],wp23_1[1][1],wp2_3[1][0],wp2_3[1][1]]
#        if (abs(rans[0]-rans1[0])<abs(rans[0]-rans2[0])):
#            logger.debug("rans = {0}".format([ rans[i] - rans1[i] for i in range(0,5)]))
#        else: logger.debug("rans = {0}".format([ rans[i] - rans2[i] for i in range(0,5)]))
#        logger.debug('pa = {}'.format(p[0]))
#        logger.debug('pb = {}'.format(p[1]))
#        logger.debug('p1 = {}'.format(p[2]))
#        logger.debug('p2 = {}'.format(p[3]))
#        logger.debug('p3 = {}'.format(p[4]))
        # The matrix element
        #lome = sherpa.ME2(p)
        #dxs = 5.*lome*wsum
        s12 = 2.*Dot(p[0],p[1])
        s13 = 2.*Dot(p[0],p[2])
        s23 = 2.*Dot(p[1],p[2])
        R = (s12+s13+s23)/s13#+s13/s23+2.*s12*(s12+s13+s23)/(s13*s23))
#        cpl = 8.*m.pi*self.alphas(mu*mu)*4./3.
        dxs = R*wsum
        logger.debug('wsum {}'.format(wsum))
#        logger.debug('lome {}'.format(lome))
        logger.debug('dxs {}'.format(dxs))
        return np.array(wsum*s13)

    def GenerateLOPoint(self,rans):
        lo = self.GeneratePoint(rans) + 1e-8
#        logger.debug('lo {}'.format(lo))
#        return ( lo[0], lo[1] )
        return lo

from qcd import AlphaS
from durham import Analysis
from comix import Comix
from integrator import *

# build and run the generator

import sys, logging
logging.basicConfig()
#logging.getLogger('eejjj').setLevel(logging.DEBUG)
#logging.getLogger('channels').setLevel(logging.DEBUG)

alphas = AlphaS(91.1876,0.118)
hardxs = eetojjj(alphas)
sherpa = Comix([11,-11],[1,-1,21])

#event, weight = hardxs.GenerateLOPoint(100)

def func(x):
    return tf.stop_gradient(tf.py_function(hardxs.GenerateLOPoint,[x],tf.float32))

#def func2(x):
#    return tf.stop_gradient((1-x[:,0])*x[:,0])

import tensorflow as tf
from tensorflow.python import debug as tf_debug

integrator = Integrator(func, 3*3-4, mode='linear')
integrator.make_optimizer(nsamples=100)
with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
#    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
#    print(np.shape(sess.run(func(np.random.random((1000,5))))))
    integrator.optimize(sess,epochs=100,printout=10)
    print(integrator.integrate(sess,1000))

#import vegas
#
#def func2(x):
#    x = np.reshape(x,(1,5))
#    return hardxs.GenerateLOPoint(x)
#
#integrand = vegas.Integrator((3*3-4)*[[0,1]])
#result = integrand(hardxs.GenerateLOPoint)
#print(result.summary())

fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,5))
ax1 = integrator.plot_loss(ax1)
ax2 = integrator.plot_integral(ax2)
ax3 = integrator.plot_variance(ax3)
plt.show() 
