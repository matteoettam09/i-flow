import Foam as fm
import math as m

class Function:
    def __init__(self):
        self.points = []
        self.weights = []
        self.plot = False
    def __call__(self,x):
        dx1, dy1, w1=0.25, 0.25, 1./0.004
        dx2, dy2, w2=0.75, 0.75, 1./0.004
        weight=m.exp(-w1*((x[0]-dx1)**2+(x[1]-dy1)**2)) \
                +m.exp(-w2*((x[0]-dx2)**2+(x[1]-dy2)**2))
        if self.plot:
            self.points.append(x)
            self.weights.append(weight)
        return weight

func = Function()

integrator = fm.Foam()
integrator.SetDimension(2)
integrator.SetNCells(10000)
integrator.SetNOpt(1000)
integrator.SetNMax(2000000)
integrator.SetError(5.0e-4)
integrator.Initialize()
integrator.Integrate(func)

import matplotlib.pyplot as plt
import corner
import numpy as np

func.plot = True
for i in range(1000000):
    wgt = integrator.Point()
    func.weights[-1] = func.weights[-1]*wgt
hist2d_kwargs={'smooth':3,'plot_datapoints':False}
figure = corner.corner(func.points,weights=func.weights,bins=100,**hist2d_kwargs)
plt.savefig('integral.pdf')
plt.close()
figure = corner.corner(func.points,bins=100,**hist2d_kwargs)
plt.savefig('sampling.pdf')
plt.close()
plt.hist(func.weights,bins=np.logspace(-2,2,100))
plt.xscale('log')
plt.yscale('log')
plt.savefig('weights.pdf')
plt.close()
