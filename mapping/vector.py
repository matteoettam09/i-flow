
import numpy as np

def Vector4(E=np.zeros(1),Px=np.zeros(1),Py=np.zeros(1),Pz=np.zeros(2)):
    return np.array([E,Px,Py,Pz]).T

def Dot(pi,pj):
    return  pi[:,0]*pj[:,0]-pi[:,1]*pj[:,1]-pi[:,2]*pj[:,2]-pi[:,3]*pj[:,3] 

def Mass2(p):
    return Dot(p,p)

def Mass(p):
    return np.sqrt(Dot(p,p))

def Momentum2(p):
    return p[:,1]*p[:,1]+p[:,2]*p[:,2]+p[:,3]*p[:,3]

def Momentum(p):
    return np.sqrt(Momentum2(p))

def PT2(p):
    return p[:,1]**2+p[:,2]**2

def PT(p):
    return np.sqrt(PT2(p))

def Theta(p):
    return np.acos(p[:,3]/Momentum(p))

def Phi(p):
    return np.where(np.logical_and(p[:,1]==0,p[:,2]==0),
                    np.zeros_like(p[:,1]),
                    np.atan2(p[:,2],p[:,1]))

def Cross(pi,pj):
    return Vector4(np.zeros_like(pi[:,1]),
                   pi[:,2]*pj[:,3]-pi[:,3]*pj[:,2],
                   pi[:,3]*pj[:,1]-pi[:,1]*pj[:,3],
                   pi[:,1]*pj[:,2]-pi[:,2]*pj[:,1])

def Boost(p,v):
    rsq = Mass(p)
    v0 = Dot(p,v)/rsq
    c1 = (v[:,0]+v0)/(rsq+p[:,0])
    return Vector4(v0,
                   v[:,1]-c1*p[:,1],
                   v[:,2]-c1*p[:,2],
                   v[:,3]-c1*p[:,3])

def BoostBack(p,v):
    rsq = Mass(p)
    v0 = (p[:,0]*v[:,0]+p[:,1]*v[:,1]+p[:,2]*v[:,2]+p[:,3]*v[:,3])/rsq
    c1 = (v[:,0]+v0)/(rsq+p[:,0])
    return Vector4(v0,
                   v[:,1]+c1*p[:,1],
                   v[:,2]+c1*p[:,2],
                   v[:,3]+c1*p[:,3])
