
import tensorflow as tf

def Vector4(E=0,Px=0,Py=0,Pz=0):
    return tf.transpose(tf.convert_to_tensor([E,Px,Py,Pz],dtype=tf.float64))

def Dot(pi,pj):
    return  pi[:,0]*pj[:,0]-pi[:,1]*pj[:,1]-pi[:,2]*pj[:,2]-pi[:,3]*pj[:,3] 

def Mass2(p):
    return tf.maximum(Dot(p,p),1e-7)

def Mass(p):
    return tf.sqrt(Mass2(p))

def Momentum2(p):
    return p[:,1]*p[:,1]+p[:,2]*p[:,2]+p[:,3]*p[:,3]

def Momentum(p):
    return tf.sqrt(Momentum2(p))

def PT2(p):
    return p[:,1]**2+p[:,2]**2

def PT(p):
    return tf.sqrt(PT2(p))

def Theta(p):
    return tf.acos(p[:,3]/Momentum(p))

def Phi(p):
    return tf.where(tf.logical_and(p[:,1]==0,p[:,2]==0),
                    tf.zeros_like(p[:,1]),
                    tf.atan2(p[:,2],p[:,1]))

def Cross(pi,pj):
    return Vector4(tf.zeros_like(pi[:,1]),
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
