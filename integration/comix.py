
from mpi4py import MPI
import sys, os
sys.path.append('/home/isaacson/Programs/lib/python3.7/site-packages')
argv=[]#,'SHERPA_LDADD=ModelMain ToolsOrg ToolsPhys ToolsMath PDF']
import Sherpa
import numpy as np

import logging
logger = logging.getLogger('eejjj')

class Comix:

    def __init__(self,flin,flout):
        f = open("Run.dat","w")
        f.write("(run){{\n SHERPA_LDADD ModelMain ToolsOrg ToolsPhys ToolsMath PDF Zfunctions\n\
  OUTPUT 15;\n\
  ME_SIGNAL_GENERATOR Comix;\n\
  EVENTS 0; GENERATE_RESULT_DIRECTORY -1;\n\
  BEAM_1 11 250; BEAM_2 -11 250;\n\
  SCALES VAR{{Abs2(p[0]+p[1])}};\n\
  PDF_SET None;\n\
  WIDTH[24] = 0;\n\
#  KFACTOR FASTJET[A:kt]{{min(MU_{3}2,1)}};\n\
}}(run);\n(processes){{\n\
#  Process {0} -> {1};\n\
  Process 11 -11 -> 6[a] -6[b];\n\
  Decay 6[a] -> 24 5; Decay -6[b] -> -24 -5;\n\
#  ME_Generator Amegic; \n\
#  Order (*,2);\n\
  End process;\n\
}}(processes);\n\
(selector){{\n\
#  FastjetFinder kt {2} 5 0 0;\n\
}}(selector);\n".format(" ".join([str(fl) for fl in flin]),\
                        " ".join([str(fl) for fl in flout]),\
                        len(flout),len(flout)-1))
        f.close()
        self.sherpa = Sherpa.Sherpa()
        print(argv)
        self.sherpa.InitializeTheRun(1,argv)
        self.process = Sherpa.MEProcess(self.sherpa)
	# for i in flin: self.process.AddInFlav(i);
	# for i in flout: self.process.AddInFlav(i);
        self.process.Initialize();

    def ME2(self,p):
        me2 = np.zeros(len(p[0]))
        for i in range(0,len(p[0])):
            for j in range(0,len(p)):
                self.process.SetMomentum(j,p[j,i,0],p[j,i,1],p[j,i,2],p[j,i,3])
            logger.debug(p[:,i])
            me2[i] = self.process.CSMatrixElement()
        return me2
