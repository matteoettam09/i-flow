
import numpy as np
import tensorflow as tf
import math as m
from absl import logging

from vector import *
#from particle import Particle

class SChannelDecay:

    def __init__(self,alpha=0):
        self.alpha = alpha
        self.eps = 1e-4

    def Lambda(self,a,b,c):
        return tf.maximum((a-b-c)**2-4*b*c,1e-7)

    def _find_axes(self,p):
        return Vector4(tf.zeros_like(p[:,0]),tf.zeros_like(p[:,1]),tf.zeros_like(p[:,2]),tf.ones_like(p[:,3])), \
            Vector4(tf.zeros_like(p[:,0]),tf.zeros_like(p[:,1]),tf.ones_like(p[:,2]),tf.zeros_like(p[:,3])), \
            Vector4(tf.zeros_like(p[:,0]),tf.ones_like(p[:,1]),tf.zeros_like(p[:,2]),tf.zeros_like(p[:,3]))

#        pl = tf.where(tf.logical_and(p[:,1]==0,tf.logical_and(p[:,2]==0,p[:,3] == 0))[:,np.newaxis],
#                Vector4(np.zeros_like(p[:,1]),np.zeros_like(p[:,1]),np.zeros_like(p[:,1]),np.ones_like(p[:,1])),
#                p/np.where(Momentum(p)==np.zeros_like(p[:,1]),np.ones_like(p[:,1]),Momentum(p))[:,np.newaxis])
#        pl[:,0] = np.zeros_like(pl[:,1])
#        pt1 = Cross(pl,Vector4(np.zeros_like(pl[:,0]),np.ones_like(pl[:,1]),np.zeros_like(pl[:,2]),np.zeros_like(pl[:,3])))
#        pt1 = pt1/Momentum(pt1)[:,np.newaxis]
#        pt2 = Cross(pt1,pl)
#        pt2 = pt2/Momentum(pt2)[:,np.newaxis]
#        return pl, pt1, pt2

    def GeneratePoint(self,p,s1,s2,rans):
        logging.debug("IsotropicPoint: {")
        logging.debug("  rans = {0}".format(rans))
        ecm = Mass(p)

        ct = 1.-(((1-rans[:,0])*self.eps**(1-self.alpha) 
            + rans[:,0]*((2.-self.eps)**(1-self.alpha))) ** (1./(1-self.alpha)))

        st = tf.sqrt(1.-ct*ct)
        phi = 2.*m.pi*rans[:,1]

        pl, pt1, pt2 = self._find_axes(p)

        ps = tf.sqrt(self.Lambda(Mass2(p),s1,s2))/(2.*ecm)
        p1 = (ps*st*tf.cos(phi))[:,tf.newaxis]*pt1 \
             +(ps*st*tf.sin(phi))[:,tf.newaxis]*pt2 \
             +(ps*ct)[:,tf.newaxis]*pl
        p1[:,0] = (Mass2(p)+s1-s2)/(2.*ecm)
        p2 = Vector4(ecm-p1[:,0],-p1[:,1],-p1[:,2],-p1[:,3])
        p1 = BoostBack(p,p1)
        p2 = BoostBack(p,p2)
        logging.debug("  pl   = {0}".format(pl))
        logging.debug("  p_T1 = {0}".format(pt1))
        logging.debug("  p_T2 = {0}".format(pt2))
        logging.debug("  \\cos\\theta = {0}, \\phi = {1}".format(ct,phi))
        logging.debug("  p   = {0}".format(p))
        logging.debug("  p_1 = {0}".format(p1))
        logging.debug("  p_2 = {0}".format(p2))
        logging.debug("  sum = {0}".format(p-p1-p2))
        logging.debug("}")
        return p1, p2

    def GenerateWeight(self,p,s1,s2,p1,p2):
        logging.debug("IsotropicWeight: {")
        logging.debug("  p   = {0}".format(p))
        logging.debug("  p_1 = {0}".format(p1))
        logging.debug("  p_2 = {0}".format(p2))
        logging.debug("  sum = {0}".format(p-p1-p2))
        ecm2 = tf.maximum(Mass2(p),1e-7)
        q1 = Boost(p,p1)
        q2 = Boost(p,p2)

        pl, pt1, pt2 = self._find_axes(p)

#        ct = -(pl*q1)/Momentum(q1)[:,tf.newaxis]
        ct = -Dot(pl,q1)/Momentum(q1)#[:,tf.newaxis]
        phi = tf.arctan(Dot(q1,pt2)/Dot(q1,pt1))

        #if ((q1*pt1)>0): phi += m.pi
        #else:
        #    if phi<0: phi += 2.*m.pi
        phi = tf.where(Dot(q1,pt1)>0,phi+m.pi,tf.where(phi<0,phi+2*m.pi,phi))

        logging.debug("  pl   = {0}".format(pl))
        logging.debug("  p_T1 = {0}".format(pt1))
        logging.debug("  p_T2 = {0}".format(pt2))
        logging.debug("  \\cos\\theta = {0}, \\phi = {1}".format(ct,phi))

        ps = tf.sqrt(self.Lambda(Mass2(p),s1,s2))/(2.*ecm2)
        I = (1./(1-self.alpha) )* ((2.-self.eps)**(1-self.alpha)-self.eps**(1-self.alpha))
        wgt = 2.*m.pi*ps/(16.*m.pi**2)

        rans = tf.convert_to_tensor([ ((1-ct)**(1-self.alpha)-(self.eps)**(1-self.alpha))/(1-self.alpha)/I, phi/(2.*m.pi) ])

        wgt *= ((1.-ct)**self.alpha)*I
        logging.debug("  rans = {0}".format(rans))
        logging.debug("  weight = {0}".format(wgt))
        logging.debug("}")
        return wgt
        
class Propagator:
    def __init__(self,alpha = 0.5):
        self.alpha = alpha

    def GeneratePoint(self,smin,smax,ran,mass=0,width=0):
        #s = smin*(smax/smin)**ran
        #s = ran*(smax-smin) + smin
        if mass == 0:
            s = ((1-ran)*smin**(1-self.alpha) + ran*(smax**(1-self.alpha))) ** (1./(1-self.alpha))
            logging.debug("MasslessPoint: ran = {0}, s_min = {1}, s_max = {2}, s = {3}".format(ran,smin,smax,s))
        else:
            mass2 = mass**2
            mw = mass*width
            ymax = tf.arctan((smin-mass2)/mw)
            ymin = tf.arctan((smax-mass2)/mw)
            s = mass2+mw*tf.tan(ymin + ran*(ymax-ymin))
            logging.debug("MassivePoint: ran = {0}, s_min = {1}, s_max = {2}, s = {3}".format(ran,smin,smax,s))

        return s

    def GenerateWeight(self,smin,smax,p,mass=0,width=0):
        s = Mass2(p)

        if mass == 0:
            #I = tf.math.log(smax/smin)
            #I = smax-smin
            I = (1./(1-self.alpha) )* (smax**(1-self.alpha)-smin**(1-self.alpha))
            ran = tf.math.log(s/smin)/I
            #wgt = s*I/(2.*m.pi)
            #wgt = I/(2*m.pi)
            wgt = (s**self.alpha)*I/(2.*m.pi)
            logging.debug("MasslessWeight: s_min = {0}, s_max = {1}, s = {2}, ran = {3}".format(smin,smax,s,ran))

        else:
            mass2 = mass**2
            mw = mass*width
            ymax = tf.arctan((smin-mass2)/mw)
            ymin = tf.arctan((smax-mass2)/mw)
            y = tf.arctan((s-mass2)/mw)
            I = ymin - ymax
            wgt = I/(2.*m.pi)
            wgt /= (mw/((s-mass2)**2+mw**2))
            ran = -(y-ymin)/I
            logging.debug("MassiveWeight: s_min = {0}, s_max = {1}, s = {2}, ran = {3}".format(smin,smax,s,ran))


        return wgt
