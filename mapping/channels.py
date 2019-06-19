
import math as m
import logging

from vector import Vec4
from particle import Particle

logger = logging.getLogger('channels')

class SChannelDecay:

    def __init__(self,flavs,pid):
        self.fl = [ flavs[pid[0]], flavs[pid[1]], flavs[pid[2]] ]

    def Lambda(self,a,b,c):
        return (a-b-c)**2-4*b*c

    def IsotropicPoint(self,p,s1,s2,rans):
        logger.debug("IsotropicPoint: {")
        logger.debug("  rans = {0}".format(rans))
        ecm = p.M()
        ct = 2.*rans[0]-1.
        st = m.sqrt(1.-ct*ct)
        phi = 2.*m.pi*rans[1]
        if p[1]==0. and p[2]==0. and p[3]==0.:
            pl = Vec4(0.,0.,0.,1.)
        else:
            pl = p/p.P()
            pl[0] = 0.
        pt1 = pl.Cross(Vec4(0.,1.,0.,0.))
        pt1 = pt1/pt1.P()
        pt2 = pt1.Cross(pl)
        pt2 = pt2/pt2.P()
        ps = m.sqrt(self.Lambda(p.M2(),s1,s2))/(2.*ecm)
        p1 = ps*st*m.cos(phi)*pt1+ps*st*m.sin(phi)*pt2+ps*ct*pl
        p1[0] = (p.M2()+s1-s2)/(2.*ecm)
        p2 = Vec4(ecm-p1[0],-p1.px,-p1.py,-p1.pz)
        p1 = p.BoostBack(p1)
        p2 = p.BoostBack(p2)
        logger.debug("  pl   = {0}".format(pl))
        logger.debug("  p_T1 = {0}".format(pt1))
        logger.debug("  p_T2 = {0}".format(pt2))
        logger.debug("  \\cos\\theta = {0}, \\phi = {1}".format(ct,phi))
        logger.debug("  p   = {0}".format(p))
        logger.debug("  p_1 = {0}".format(p1))
        logger.debug("  p_2 = {0}".format(p2))
        logger.debug("  sum = {0}".format(p-p1-p2))
        logger.debug("}")
        return [ p1, p2 ]
        
    def IsotropicWeight(self,p,s1,s2,p1,p2):
        logger.debug("IsotropicWeight: {")
        logger.debug("  p   = {0}".format(p))
        logger.debug("  p_1 = {0}".format(p1))
        logger.debug("  p_2 = {0}".format(p2))
        logger.debug("  sum = {0}".format(p-p1-p2))
        ecm = p.M()
        q1 = p.Boost(p1)
        q2 = p.Boost(p2)
        if p[1]==0. and p[2]==0. and p[3]==0.:
            pl = Vec4(0.,0.,0.,1.)
        else:
            pl = p/p.P()
            pl[0] = 0.
        pt1 = pl.Cross(Vec4(0.,1.,0.,0.))
        pt1 = pt1/pt1.P()
        pt2 = pt1.Cross(pl)
        pt2 = pt2/pt2.P()
        ct = -(pl*q1)/q1.P()
        phi = m.atan((q1*pt2)/(q1*pt1))
        if ((q1*pt1)>0): phi += m.pi
        else:
            if phi<0: phi += 2.*m.pi
        logger.debug("  pl   = {0}".format(pl))
        logger.debug("  p_T1 = {0}".format(pt1))
        logger.debug("  p_T2 = {0}".format(pt2))
        logger.debug("  \\cos\\theta = {0}, \\phi = {1}".format(ct,phi))
        rans = [ (1.+ct)/2., phi/(2.*m.pi) ]

        ps = m.sqrt(self.Lambda(p.M2(),s1,s2))/(2.*ecm)
        wgt = 4.*m.pi*ps/(16.*m.pi**2*ecm)
        logger.debug("  rans = {0}".format(rans))
        logger.debug("  weight = {0}".format(wgt))
        logger.debug("}")
        return [ wgt, rans ]
        
    def GeneratePoint(self,p,s1,s2,rans):
        return self.IsotropicPoint(p,s1,s2,rans) 

    def GenerateWeight(self,p,s1,s2,p1,p2):
        return self.IsotropicWeight(p,s1,s2,p1,p2) 

class Propagator:

    def __init__(self,flavs,pid):
        self.fl = flavs[pid]

    def MasslessPoint(self,smin,smax,ran):
        s = smin*m.pow(smax/smin,ran)
        logger.debug("MasslessPoint: ran = {0}, s_min = {1}, s_max = {2}, s = {3}".format(ran,s,smin,smax))
        return s

    def MasslessWeight(self,smin,smax,p):
        s = p.M2()
        I = m.log(smax/smin)
        ran = m.log(s/smin)/I
        wgt = I/(2.*m.pi)
        logger.debug("MasslessWeight: ran = {0}, s_min = {1}, s_max = {2}, s = {3}".format(ran,s,smin,smax))
        return [ wgt, ran ]

    def GeneratePoint(self,smin,smax,ran):
        return self.MasslessPoint(smin,smax,ran)

    def GenerateWeight(self,smin,smax,p):
        return self.MasslessWeight(smin,smax,p)
