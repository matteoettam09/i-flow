import Foam as fm
import math as m

class Function:
    def __init__(self):
        self.points = []
        self.weights = []
        self.plot = False
    def __call__(self,x):
        dx1, dy1, rr, w1, ee = 0.4, 0.6, 0.25, 1./0.004, 3.0
        weight=m.pow(x[1],ee)*m.exp(-w1*abs((x[1]-dy1)**2+(x[0]-dx1)**2-rr**2))+\
            m.pow(1.0-x[1],ee)*m.exp(-w1*abs((x[1]-1.0+dy1)**2+(x[0]-1.0+dx1)**2-rr**2))
        if self.plot:
            self.points.append(x)
            self.weights.append(weight)
        return weight

func = Function()

integrator = fm.Foam()
integrator.SetDimension(2)
integrator.SetNCells(500)
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
