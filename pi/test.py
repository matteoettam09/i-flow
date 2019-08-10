import Foam as fm
import math as m

class Function:
    def call(self,x):
        dx1, dy1, w1=0.25, 0.25, 1./0.004
        dx2, dy2, w2=0.75, 0.75, 1./0.004
        weight=m.exp(-w1*((x[0]-dx1)**2+(x[1]-dy1)**2)) \
                +m.exp(-w2*((x[0]-dx2)**2+(x[1]-dy2)**2))
        return weight

func = Function()
integrator = fm.Foam()
integrator.SetDimension(2);
integrator.SetNCells(500);
integrator.SetNOpt(1000);
integrator.SetNMax(2000000);
integrator.SetError(5.0e-4);
integrator.Initialize()
integrator.Integrate(func)


