
import math as m
import random as r

from vector import Vec4
from particle import Particle
from channels import SChannelDecay, Propagator
from durham import Algorithm

import logging
logger = logging.getLogger('eejjj')

class eetojjj:

    def __init__(self,alphas,ecms=91.2):
        self.alphas = alphas
        self.ecms = ecms
        self.MZ2 = pow(91.1876,2.)
        self.GZ2 = pow(2.4952,2.)
        self.alpha = 1./128.802
        self.sin2tw = 0.22293
        self.duralg = Algorithm()

    def ME2(self,fl,s,t):
        qe = -1.
        ae = -0.5
        ve = ae - 2.*qe*self.sin2tw;
        qf = 2./3. if fl in [2,4] else -1./3.
        af = 0.5 if fl in [2,4] else -0.5
        vf = af - 2.*qf*self.sin2tw;
        kappa = 1./(4.*self.sin2tw*(1.-self.sin2tw))
        chi1 = kappa * s * (s-self.MZ2)/(pow(s-self.MZ2,2.) + self.GZ2*self.MZ2);
        chi2 = pow(kappa * s,2.)/(pow(s-self.MZ2,2.) + self.GZ2*self.MZ2);
        term1 = (1+pow(1.+2.*t/s,2.))*(pow(qf*qe,2.)+2.*(qf*qe*vf*ve)*chi1+(ae*ae+ve*ve)*(af*af+vf*vf)*chi2)
        term2 = (1.+2.*t/s)*(4.*qe*qf*ae*af*chi1+8.*ae*ve*af*vf*chi2)
        return pow(4.*m.pi*self.alpha,2.)*3.*(term1+term2)

    def GeneratePoint(self):
        pa = Vec4(self.ecms/2,0,0,self.ecms/2)
        pb = Vec4(self.ecms/2,0,0,-self.ecms/2)
        fl = r.randint(1,5)
        # The phase space generation
        prop = Propagator(flavs,fl)
        decay = SChannelDecay(flavs,[fl,fl,21])
        rans = [ r.random() for i in range(0,5) ]
        if r.random()<0.5:
            logger.debug("(13)2")
            s13 = prop.GeneratePoint(1.e-3,self.ecms**2,rans[0])
            p13_2 = decay.GeneratePoint(pa+pb,s13,0.,[ rans[1], rans[2] ])
            p1_3 = decay.GeneratePoint(p13_2[0],0.,0.,[ rans[3], rans[4] ])
            p = [ p1_3[0], p13_2[1], p1_3[1] ]
        else:
            logger.debug("1(23)")
            s23 = prop.GeneratePoint(1.e-3,self.ecms**2,rans[0])
            p23_1 = decay.GeneratePoint(pa+pb,s23,0.,[ rans[1], rans[2] ])
            p2_3 = decay.GeneratePoint(p23_1[0],0.,0.,[ rans[3], rans[4] ])
            p = [ p23_1[1], p2_3[0], p2_3[1] ]            
        logger.debug("p_0 = {0}, m2 = {1}".format(p[0],p[0].M2()))
        logger.debug("p_1 = {0}, m2 = {1}".format(p[1],p[1].M2()))
        logger.debug("p_2 = {0}, m2 = {1}".format(p[1],p[2].M2()))
        logger.debug("sum: {0}".format(pa+pb-(p[0]+p[1]+p[2])))
        # The phase space weight
        p13 = p[0]+p[2]
        ws13 = prop.GenerateWeight(1.e-3,self.ecms**2,p13)
        wp13_2 = decay.GenerateWeight(pa+pb,p13.M2(),0.,p[0]+p[2],p[1])
        wp1_2 = decay.GenerateWeight(p[0]+p[2],0.,0.,p[0],p[2])
        logger.debug("weights {0} {1} {2}".format(ws13,wp13_2,wp1_2))
        p23 = p[1]+p[2]
        ws23 = prop.GenerateWeight(1.e-3,self.ecms**2,p23)
        wp23_1 = decay.GenerateWeight(pa+pb,p23.M2(),0.,p[1]+p[2],p[0])
        wp2_3 = decay.GenerateWeight(p[1]+p[2],0.,0.,p[1],p[2])
        logger.debug("weights {0} {1} {2}".format(ws23,wp23_1,wp2_3))
        wsum = wp13_2[0]*ws13[0]*wp1_2[0]+wp23_1[0]*ws23[0]*wp2_3[0]
        logger.debug("total weight {0}".format(wsum))
        rans1 = [ws13[1],wp13_2[1][0],wp13_2[1][1],wp1_2[1][0],wp1_2[1][1]]
        rans2 = [ws23[1],wp23_1[1][0],wp23_1[1][1],wp2_3[1][0],wp2_3[1][1]]
        if (abs(rans[0]-rans1[0])<abs(rans[0]-rans2[0])):
            logger.debug("rans = {0}".format([ rans[i] - rans1[i] for i in range(0,5)]))
        else: logger.debug("rans = {0}".format([ rans[i] - rans2[i] for i in range(0,5)]))
        # The matrix element
        lome = self.ME2(fl,(pa+pb).M2(),(pa-p[0]).M2())
        dxs = 5.*lome*3.89379656e8/(8.*m.pi)/(2.*pow(self.ecms,2))
        event = [
            Particle(-11,-pa),
            Particle(11,-pb),
            Particle(fl,p[0],[2,0]),
            Particle(-fl,p[1],[0,1]),
            Particle(21,p[2],[1,2]) ]
        kt2 = self.duralg.Cluster(event)
        mu = self.ecms
        if kt2[-1]<1e-3:
            dxs = 0.
        else:
            s12 = 2.*p[0]*p[1]
            s13 = 2.*p[0]*p[2]
            s23 = 2.*p[1]*p[2]
            R = lome*(s23/s13+s13/s23+2.*s12*(s12+s13+s23)/(s13*s23))
            cpl = 8.*m.pi*self.alphas(mu*mu)*4./3.
            dxs = dxs*cpl*R
        return ( event, dxs, lome )

    def GenerateLOPoint(self):
        lo = self.GeneratePoint()
        return ( lo[0], lo[1] )

from qcd import AlphaS
from durham import Analysis

# build and run the generator

import sys, logging
logging.basicConfig()
logging.getLogger('eejjj').setLevel(logging.DEBUG)
#logging.getLogger('channels').setLevel(logging.DEBUG)

alphas = AlphaS(91.1876,0.118)
hardxs = eetojjj(alphas)
jetrat = Analysis()

flavs = { 21:[0,0,2] }
for i in range(-5,0):
    flavs[i] = [0,0,1]
for i in range(1,6):
    flavs[i] = [0,0,1]

r.seed(123456)
for i in range(100000):
    event, weight = hardxs.GenerateLOPoint()
    sys.stdout.write('\rEvent {0}'.format(i))
    sys.stdout.flush()
    jetrat.Analyze(event,weight)
jetrat.Finalize("Analysis")
print ""
