import numpy as np
import time

from vector import *
from channels import SChannelDecay, Propagator

import logging
logger = logging.getLogger('eejjj')


start_time = time.time()

class eetojjj:

    def __init__(self,alphas,ecms=91.2):
        self.ecms = ecms
        self.prop = Propagator()
        # self.decay = SChannelDecay()
        self.decay = SChannelDecay(0.99)
        self.decayIso = SChannelDecay(0.)
        self.channel = 1
        self.channels={}
        self.channels[0] = self.Channel1
        self.channels[1] = self.Channel2

    def Channel1(self,rans,pa,pb):
        #print("Ch1")
        s13 = self.prop.GeneratePoint(1e-3,self.ecms**2,rans[:,0])
        #p13, p2 = self.decay.GeneratePoint(pa+pb,s13,0.0,np.array([rans[:,1], rans[:,2]]).T)
        p13, p2 = self.decayIso.GeneratePoint(pa+pb,s13,0.0,np.array([rans[:,1], rans[:,2]]).T)
        p1, p3 = self.decay.GeneratePoint(p13,0.0,0.0,np.array([rans[:,3], rans[:,4]]).T)

        return p1, p2, p3

    def Channel2(self,rans,pa,pb):
        #print("Ch2")
        s23 = self.prop.GeneratePoint(1e-3,self.ecms**2,rans[:,0])
        #p23, p1 = self.decay.GeneratePoint(pa+pb,s23,0.0,np.array([rans[:,1], rans[:,2]]).T)
        p23, p1 = self.decayIso.GeneratePoint(pa+pb,s23,0.0,np.array([rans[:,1], rans[:,2]]).T)
        p2, p3 = self.decay.GeneratePoint(p23,0.0,0.0,np.array([rans[:,3], rans[:,4]]).T)

        return p1, p2, p3

    def ChannelIso(self,rans,pa,pb):
        s12 = self.prop.GeneratePoint(0.,self.ecms**2,rans[:,2])
        p1, p2 = self.decay.GeneratePoint(pa+pb,s12,0.0,np.array([rans[:,0], rans[:,1]]).T)

        return p1, p2

    def GeneratePoint(self,rans):
        #print("True function")
        #print("rans: "+str(rans))
        rans=np.minimum(rans,1.-5e-8)
        #print("rans corrected: "+str(rans))
        logger.debug("  rans = {0}".format(rans))
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

        channels = np.random.randint(0,len(self.channels),(rans.shape[0]))
        unique, counts = np.unique(channels, return_counts = True)
        counts = np.cumsum(counts)
        channels = dict(zip(unique, counts))
        #channels={0:rans.shape[0]//2,1:rans.shape[0]}
#        p1, p2 = self.ChannelIso(rans,pa,pb)
        p1 = []
        p2 = []
        p3 = []
        position_old = 0
        for channel, position in channels.items():
            #print("channel: "+str(channel))
            #print("position: "+str(position))
            p1tmp, p2tmp, p3tmp = self.channels[channel](rans[position_old:position],pa[position_old:position],pb[position_old:position])
            p1.append(p1tmp)
            p2.append(p2tmp)
            p3.append(p3tmp)
            position_old = position

        p1 = np.concatenate(p1)
        p2 = np.concatenate(p2)
        p3 = np.concatenate(p3)

        #if self.channel == 0:
        #    p1, p2, p3 = self.Channel1(rans,pa,pb)
        #else:
        #    p1, p2, p3 = self.Channel2(rans,pa,pb)
        ##if channel == 1:
        ##    p1, p2, p3 = self.Channel1(rans,pa,pb)
        ##else:
        ##    p1, p2, p3 = self.Channel2(rans,pa,pb)

        p13 = p1+p3
        ws13 = self.prop.GenerateWeight(1e-3,self.ecms**2,p13)
        #wp13_2 = self.decay.GenerateWeight(pa+pb,Mass2(p13),0.0,p13,p2)
        wp13_2 = self.decayIso.GenerateWeight(pa+pb,Mass2(p13),0.0,p13,p2)
        wp1_3 = self.decay.GenerateWeight(p1+p3,0.0,0.0,p1,p3)

        p23 = p2+p3
        ws23 = self.prop.GenerateWeight(1e-3,self.ecms**2,p23)
        #wp23_1 = self.decay.GenerateWeight(pa+pb,Mass2(p23),0.0,p23,p1)
        wp23_1 = self.decayIso.GenerateWeight(pa+pb,Mass2(p23),0.0,p23,p1)
        wp2_3 = self.decay.GenerateWeight(p23,0.0,0.0,p2,p3)

        wsum = 1./((0.5/(wp13_2*ws13*wp1_3)) + (0.5/(wp23_1*ws23*wp2_3)))
        #lome = sherpa.ME2(np.array([pa,pb,p3,p1,p2]))
        lome = np.array(sherpa.process.CSMatrixElementVec(np.array([pa,pb,p3,p1,p2])))
        """
        s12 = 2.*Dot(p1,p2)
        s13 = 2.*Dot(p1,p3)
        s23 = 2.*Dot(p2,p3)
        #lome = (s23/s13+s13/s23+2.*s12*(s12+s13+s23)/(s13*s23))
        lome = (2.*(s12)/(s13*s23))
        temp = s13*s23 /(s12+s13+s23)
        lome *= np.minimum(temp,np.ones_like(temp))
        """    
        dxs = (lome*wsum*3.89379656e8/(2.*self.ecms**2))+1e-8
        return dxs

    """
    def GeneratePointCh(self,rans,channel):
        #print("Generate Point Channel "+str(channel))
        rans=np.minimum(rans,1.-5e-8)
        logger.debug("  rans = {0}".format(rans))
        pa = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     self.ecms/2*np.ones(np.shape(rans)[0]))
        pb = Vector4(self.ecms/2*np.ones(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     np.zeros(np.shape(rans)[0]),
                     -self.ecms/2*np.ones(np.shape(rans)[0]))

        
        #channels = np.random.randint(0,len(self.channels),(rans.shape[0]))
        #unique, counts = np.unique(channels, return_counts = True)
        #counts = np.cumsum(counts)
        #channels = dict(zip(unique, counts))
        #channels={0:rans.shape[0]//2,1:rans.shape[0]}
#        p1, p2 = self.ChannelIso(rans,pa,pb)
        #p1 = []
        #p2 = []
        #p3 = []
        #position_old = 0
        #for channel, position in channels.items():
            #print("channel: "+str(channel))
            #print("position: "+str(position))
            #p1tmp, p2tmp, p3tmp = self.channels[channel](rans[position_old:position],pa[position_old:position],pb[position_old:position])
            #p1.append(p1tmp)
            #p2.append(p2tmp)
            #p3.append(p3tmp)
            #position_old = position

        p1, p2, p3 = self.channels[channel](rans,pa,pb)

        p13 = p1+p3
        ws13 = self.prop.GenerateWeight(1,self.ecms**2,p13)
        wp13_2 = self.decay.GenerateWeight(pa+pb,Mass2(p13),0.0,p13,p2)
        wp1_3 = self.decay.GenerateWeight(p1+p3,0.0,0.0,p1,p3)

        p23 = p2+p3
        ws23 = self.prop.GenerateWeight(1,self.ecms**2,p23)
        wp23_1 = self.decay.GenerateWeight(pa+pb,Mass2(p23),0.0,p23,p1)
        wp2_3 = self.decay.GenerateWeight(p23,0.0,0.0,p2,p3)

        wsum = 1./((0.5/(wp13_2*ws13*wp1_3)) + (0.5/(wp23_1*ws23*wp2_3)))
        #lome = sherpa.ME2(np.array([pa,pb,p3,p1,p2]))
        lome = np.array(sherpa.process.CSMatrixElementVec(np.array([pa,pb,p3,p1,p2])))
        dxs = (lome*wsum*3.89379656e8/(2.*self.ecms**2))+1e-8
        return dxs
    
    def GeneratePoint_train(self,rans):
        #print("Training function")
        Ch1 = self.GeneratePointCh(rans,0)
        Ch2 = self.GeneratePointCh(rans,1)
        return np.average(np.array([Ch1,Ch2]),axis=0)
    """
if __name__ == '__main__':
    from qcd import AlphaS
    from comix import Comix
    from integrator import *

    import logging
    logging.basicConfig()
    #logging.getLogger('channels').setLevel(logging.DEBUG)
    #logging.getLogger('eejjj').setLevel(logging.DEBUG)

    alphas = AlphaS(91.1876,0.118)
    hardxs = eetojjj(alphas)
    sherpa = Comix([11,-11],[1,-1,21])

    def func_int(x):
        return tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x],tf.float32))
    def func_train(x):
        return tf.stop_gradient(tf.py_function(hardxs.GeneratePoint_train,[x],tf.float32))
##        nbatchs = 10
##        results = []
##        batch_size = x.shape[0]//nbatchs
##        for i in range(nbatchs):
##            hardxs.channel = np.random.randint(0,2)
##            print("channel ran: "+str(hardxs.channel))
##            results.append(tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x[i*batch_size:(i+1)*batch_size]],tf.float32)))
##        return tf.concat(results,0)
#        hardxs.channel = 1
#        channel1 = tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x[:x.shape[0]//2]],tf.float32))
#        hardxs.channel = 2
#        channel2 = tf.stop_gradient(tf.py_function(hardxs.GeneratePoint,[x[x.shape[0]//2:]],tf.float32))
#        return tf.concat([channel1,channel2],0)


    import corner
    import os.path
    if os.path.exists('matrix.pdf')==False:
        print(" * * * Plotting Matrixelement * * * ")
        x = np.random.random((100000,5))
        #hardxs.channel = 1
        p = hardxs.GeneratePoint(x)

        #x2 = np.random.random((100000,5))
        #hardxs.channel = 2
        #p2 = hardxs.GeneratePoint(x2)

        #x = np.concatenate([x1,x2])
        #p = np.concatenate([p1,p2])
        #p = p/np.mean(p)
        figure = corner.corner(x, labels=[r'$x_0$',r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$'], weights=p,show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig('matrix.pdf')
    else:
        print(" * * * Skipping Matrixelement plot * * * ")
    
    import tensorflow as tf
    from tensorflow.python import debug

    integrator = Integrator(func_int, 5, mode='linear_blob',nbins=16)
    integrator.make_optimizer(nsamples=1000, learning_rate=5e-4)

    with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
        #sess=debug.LocalCLIDebugWrapperSession(sess)
        #sess=debug.TensorBoardDebugWrapperSession(sess, 'Triton:6064',send_traceback_and_source_code=False)
        sess.run(tf.global_variables_initializer())
        integrator.optimize(sess,epochs=500,printout=25)
        #integrator.func=func_int
        print(integrator.integrate(sess,25000))

    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(16,5))
    ax1 = integrator.plot_loss(ax1)
    ax2 = integrator.plot_integral(ax2)
    ax3 = integrator.plot_variance(ax3)
    plt.savefig('loss.pdf') 



Time = time.time() - start_time
Hours = Time//3600
Minutes = (Time - 3600*Hours)//60
Seconds = (Time - 3600*Hours - 60*Minutes)

print("Done")
print("--- %(STD)s hours, %(MIN)s minutes and %(SEC).2f seconds ---" %{'STD':Hours,'MIN':Minutes, 'SEC':Seconds})
