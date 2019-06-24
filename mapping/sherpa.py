
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

    def GeneratePoint(self,batch):
        pa = Vector4(self.ecms/2*np.ones(batch),np.zeros(batch),np.zeros(batch),self.ecms/2*np.ones(batch))
        pb = Vector4(self.ecms/2*np.ones(batch),np.zeros(batch),np.zeros(batch),-self.ecms/2*np.ones(batch))
        # The phase space generation
        rans = np.random.random((batch,5))
        if np.random.random()<0.5:
            logger.debug("(13)2")
            logger.debug(pa+pb)
            s13 = self.prop.GeneratePoint(1.e-3,self.ecms**2,rans[:,0])
            p13_2 = self.decay.GeneratePoint(pa+pb,s13,0.,np.array([rans[:,1], rans[:,2]]).T)
            p1_3 = self.decay.GeneratePoint(p13_2[0],0.,0.,np.array([rans[:,3], rans[:,4]]).T)
            p = np.array([ p1_3[0], p13_2[1], p1_3[1] ])
        else:
            logger.debug("1(23)")
            s23 = self.prop.GeneratePoint(1.e-3,self.ecms**2,rans[:,0])
            p23_1 = self.decay.GeneratePoint(pa+pb,s23,0.,np.array([ rans[:,1], rans[:,2] ]).T)
            p2_3 = self.decay.GeneratePoint(p23_1[0],0.,0.,np.array([ rans[:,3], rans[:,4] ]).T)
            p = np.array([ p23_1[1], p2_3[0], p2_3[1] ])
        logger.debug("p_0 = {0}, m2 = {1}".format(p[0],Mass2(p[:,0])))
        logger.debug("p_1 = {0}, m2 = {1}".format(p[1],Mass2(p[:,1])))
        logger.debug("p_2 = {0}, m2 = {1}".format(p[2],Mass2(p[:,2])))
        logger.debug("sum: {0}".format(pa+pb-(p[0]+p[1]+p[2])))
        # The phase space weight
        p13 = p[0]+p[2]
        ws13 = self.prop.GenerateWeight(1.e-3,self.ecms**2,p13)
        wp13_2 = self.decay.GenerateWeight(pa+pb,Mass2(p13),0.,p[0]+p[2],p[1])
        wp1_2 = self.decay.GenerateWeight(p[0]+p[2],0.,0.,p[0],p[2])
        logger.debug("weights {0} {1} {2}".format(ws13,wp13_2,wp1_2))
        p23 = p[1]+p[2]
        ws23 = self.prop.GenerateWeight(1.e-3,self.ecms**2,p23)
        wp23_1 = self.decay.GenerateWeight(pa+pb,Mass2(p23),0.,p[1]+p[2],p[0])
        wp2_3 = self.decay.GenerateWeight(p[1]+p[2],0.,0.,p[1],p[2])
        logger.debug("weights {0} {1} {2}".format(ws23,wp23_1,wp2_3))
        wsum = wp13_2[0]*ws13[0]*wp1_2[0]+wp23_1[0]*ws23[0]*wp2_3[0]
        logger.debug("total weight {0}".format(wsum))
        rans1 = [ws13[1],wp13_2[1][0],wp13_2[1][1],wp1_2[1][0],wp1_2[1][1]]
        rans2 = [ws23[1],wp23_1[1][0],wp23_1[1][1],wp2_3[1][0],wp2_3[1][1]]
#        if (abs(rans[0]-rans1[0])<abs(rans[0]-rans2[0])):
#            logger.debug("rans = {0}".format([ rans[i] - rans1[i] for i in range(0,5)]))
#        else: logger.debug("rans = {0}".format([ rans[i] - rans2[i] for i in range(0,5)]))
        # The matrix element
        Process.SetMomentum(0,pa[0],pa[1],pa[2],pa[3])
        Process.SetMomentum(1,pb[0],pb[1],pb[2],pb[3])
        for i in range(0,3):
            Process.SetMomentum(i+2,p[i][0],p[i][1],p[i][2],p[i][3])
        lome = Process.CSMatrixElement()
        dxs = 5.*lome*wsum
        event = [
            Particle(-11,-pa),
            Particle(11,-pb),
            Particle(1,p[0],[2,0]),
            Particle(-1,p[1],[0,1]),
            Particle(21,p[2],[1,2]) ]
        kt2 = self.duralg.Cluster(event)
        mu = self.ecms
        if kt2[-1]<1e-3:
            dxs = 0.
        return ( event, dxs, lome )

    def GenerateLOPoint(self,batch):
        lo = self.GeneratePoint(batch)
        return ( lo[0], lo[1] )

from qcd import AlphaS
from durham import Analysis

# build and run the generator

from mpi4py import MPI
import sys, os
sys.path.append('/home/stefan/hep/sherpa/rel-2-2-7/lib/python2.7/site-packages')
argv=['Sherpa','SHERPA_LDADD=ModelMain ToolsOrg ToolsPhys ToolsMath PDF']
import Sherpa

Generator=Sherpa.Sherpa()
Generator.InitializeTheRun(len(argv),argv)
Process=Sherpa.MEProcess(Generator)
Process.AddInFlav(11);
Process.AddInFlav(-11);
Process.AddOutFlav(1);
Process.AddOutFlav(-1);
Process.AddOutFlav(21);
Process.Initialize();

import sys, logging
logging.basicConfig()
#logging.getLogger('eejjj').setLevel(logging.DEBUG)
#logging.getLogger('channels').setLevel(logging.DEBUG)

alphas = AlphaS(91.1876,0.118)
hardxs = eetojjj(alphas)
jetrat = Analysis()

np.random.seed(123456)
event, weight = hardxs.GenerateLOPoint(100)
sys.stdout.write('\rEvent {0}'.format(i))
sys.stdout.flush()
jetrat.Analyze(event,weight)
jetrat.Finalize("Analysis")
print("")
