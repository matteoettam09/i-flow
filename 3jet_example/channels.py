
import numpy as np
import math as m
import logging

from vector import *
from particle import Particle

logger = logging.getLogger('channels')

class SChannelDecay:
    def __init__(self,alpha):
        self.alpha=alpha
        self.eps=1e-3

    def Lambda(self,a,b,c):
        return (a-b-c)**2-4*b*c

    def _find_axes(self,p):
        pl = np.where(np.logical_and(p[:,1]==0,np.logical_and(p[:,2]==0,p[:,3] == 0))[:,np.newaxis],
                Vector4(np.zeros_like(p[:,1]),np.zeros_like(p[:,1]),np.zeros_like(p[:,1]),np.ones_like(p[:,1])),
                p/np.where(Momentum(p)==np.zeros_like(p[:,1]),np.ones_like(p[:,1]),Momentum(p))[:,np.newaxis])
        pl[:,0] = np.zeros_like(pl[:,1])
        pt1 = Cross(pl,Vector4(np.zeros_like(pl[:,0]),np.ones_like(pl[:,1]),np.zeros_like(pl[:,2]),np.zeros_like(pl[:,3])))
        pt1 = pt1/Momentum(pt1)[:,np.newaxis]
        pt2 = Cross(pt1,pl)
        pt2 = pt2/Momentum(pt2)[:,np.newaxis]
        return pl, pt1, pt2

    def GeneratePoint(self,p,s1,s2,rans):
        logger.debug("IsotropicPoint: {")
        logger.debug("  rans = {0}".format(rans))
        ecm = Mass(p)
        #ct = 2.*rans[:,0]-1.
        ct = 1.-(((1-rans[:,0])*self.eps**(1-self.alpha) + rans[:,0]*((2.-self.eps)**(1-self.alpha))) ** (1./(1-self.alpha)))
        st = np.sqrt(1.-ct*ct)
        phi = 2.*m.pi*rans[:,1]

        pl, pt1, pt2 = self._find_axes(p)

        ps = np.sqrt(self.Lambda(Mass2(p),s1,s2))/(2.*ecm)
        p1 = (ps*st*np.cos(phi))[:,np.newaxis]*pt1 \
             +(ps*st*np.sin(phi))[:,np.newaxis]*pt2 \
             +(ps*ct)[:,np.newaxis]*pl
        p1[:,0] = (Mass2(p)+s1-s2)/(2.*ecm)
        p2 = Vector4(ecm-p1[:,0],-p1[:,1],-p1[:,2],-p1[:,3])
        p1 = BoostBack(p,p1)
        p2 = BoostBack(p,p2)
        logger.debug("  pl   = {0}".format(pl))
        logger.debug("  p_T1 = {0}".format(pt1))
        logger.debug("  p_T2 = {0}".format(pt2))
        logger.debug("  \\cos\\theta = {0}, \\phi = {1}".format(ct,phi))
        logger.debug("  p   = {0}".format(p))
        logger.debug("  p_1 = {0}".format(p1))
        logger.debug("  p_2 = {0}".format(p2))
        logger.debug("  sum = {0}".format(p-p1-p2))
        logger.debug("}")
        return p1, p2

    def GenerateWeight(self,p,s1,s2,p1,p2):
        logger.debug("IsotropicWeight: {")
        logger.debug("  p   = {0}".format(p))
        logger.debug("  p_1 = {0}".format(p1))
        logger.debug("  p_2 = {0}".format(p2))
        logger.debug("  sum = {0}".format(p-p1-p2))
        ecm2 = np.maximum(Mass2(p),1e-7)
        
        q1 = Boost(p,p1)
        q2 = Boost(p,p2)

        pl, pt1, pt2 = self._find_axes(p)

        ct = -Dot(pl,q1)/Momentum(q1)#[:,np.newaxis]
#        phi = np.arctan2((q1*pt2),(q1*pt1))

        #if ((q1*pt1)>0): phi += m.pi
        #else:
        #    if phi<0: phi += 2.*m.pi
#        phi = np.where(phi<np.zeros_like(phi),phi+2*m.pi,phi)

#        logger.debug("  pl   = {0}".format(pl))
#        logger.debug("  p_T1 = {0}".format(pt1))
#        logger.debug("  p_T2 = {0}".format(pt2))
#        logger.debug("  \\cos\\theta = {0}, \\phi = {1}".format(ct,phi))
#        rans = [ (1.+ct)/2., phi/(2.*m.pi) ]

        ps = np.sqrt(self.Lambda(Mass2(p),s1,s2))/(2.*ecm2)
        I = (1./(1-self.alpha) )* ((2.-self.eps)**(1-self.alpha)-self.eps**(1-self.alpha))
        wgt = 2.*m.pi*ps/(16.*m.pi**2)

        wgt *= ((1.-ct)**self.alpha)*I
        # old:
        #wgt = 4.*m.pi*ps/(16.*m.pi**2)
        
#        logger.debug("  rans = {0}".format(rans))
        logger.debug("  weight = {0}".format(wgt))
        logger.debug("}")
        return wgt
        
class Propagator:
    def __init__(self):
        self.alpha = 0.99
        
    def GeneratePoint(self,smin,smax,ran):
        #s = smin*(smax/smin)**ran
        #s = ran*(smax-smin) + smin
        s = ((1-ran)*smin**(1-self.alpha) + ran*(smax**(1-self.alpha))) ** (1./(1-self.alpha))
        logger.debug("MasslessPoint: ran = {0}, s_min = {1}, s_max = {2}, s = {3}".format(ran,smin,smax,s))
        return s

    def GenerateWeight(self,smin,smax,p):
        s = np.maximum(Mass2(p),1e-7)
        #I = np.log(smax/smin)
        #I = smax-smin
        I = (1./(1-self.alpha) )* (smax**(1-self.alpha)-smin**(1-self.alpha))
        #ran = np.log(s/smin)/I
        #wgt = s*I/(2.*m.pi)
        #wgt = I/(2.*m.pi)
        #s=np.maximum(s,np.zeros_like(s))
        wgt = (s**self.alpha)*I/(2.*m.pi)
        logger.debug("MasslessWeight: s_min = {0}, s_max = {1}, s = {2}".format(smin,smax,s))
        return wgt


