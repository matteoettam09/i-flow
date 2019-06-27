
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
#get_ipython().magic(u'matplotlib inline')
#from piecewise import *
from piecewiseUnet import *
import time
import corner as corner
from tensorflow.python import debug


# ## Settings

# In[2]:


NP_DTYPE = np.float32




sess = tf.InteractiveSession()
sess = debug.LocalCLIDebugWrapperSession(sess)


# In[4]:


bijectors = []
for i in range(4):
    bijectors.append(PiecewiseQuadratic(6,3,24,layer_id=i,name="PwQ"))
    bijectors.append(tfb.Permute(permutation=[1,2,3,4,5,0],name="Prm"))

bijectors.append(tfb.Permute(permutation=[2,3,4,5,0,1],name="Prm2"))


# Discard the last Permute layer

#test_bijector = tfb.Chain(list(reversed(bijectors[:-1])))
test_bijector = tfb.Chain(list(reversed(bijectors[:])))
test_bijector.forward_min_event_ndims


# In[5]:


base_dist = tfd.Uniform(low=[0.0,0.0,0.0,0.0,0.0,0.0],high=[1.0,1.0,1.0,1.0,1.0,1.0])
base_dist = tfd.Independent(distribution=base_dist,
                           reinterpreted_batch_ndims=1,
                           )
dist = tfd.TransformedDistribution(
    distribution=base_dist,
    bijector=test_bijector,
)





def step(x,x0,k=50):
    return tf.abs(0.5+0.5*tf.tanh(k*(x0-x)))

def step_np(x,x0,k=50):
    return np.abs(0.5+0.5*np.tanh(k*(x0-x)))

def dsigmaTrain(x):
    #return tf.where(tf.logical_and(x[:,0] < tf.ones_like(x[:,0])*0.9,x[:,1] < tf.ones_like(x[:,0])*0.9), (x[:,0]**2 + x[:,1]**2)/((1-x[:,0])*(1-x[:,1])), tf.zeros_like(x[:,0]))
    return (x[:,0]**2+x[:,1]**2)/((1-x[:,0])*(1-x[:,1]))*step(x[:,0],0.9)*step(x[:,1],0.9)

def dsigmaCircle(x):
    return (x[:,0]**2+x[:,1]**2)/((1.-x[:,0])*(1.-x[:,1]))*step(tf.sqrt(x[:,0]**2+x[:,1]**2),0.95)+1e-6

def dsigmaCircle_np(x):
    x0,x1=x
    return (x0**2+x1**2)/((1.-x0)*(1.-x1))*step_np(np.sqrt(x0**2+x1**2),0.95)+1e-6

def dsigmaTrue(x):
    return tf.where(tf.logical_and(x[:,0] < tf.ones_like(x[:,0])*0.9,x[:,1] < tf.ones_like(x[:,0])*0.9), (x[:,0]**2 + x[:,1]**2)/((1-x[:,0])*(1-x[:,1])), tf.zeros_like(x[:,0]))

def dsigmaTest(x):
    return 10**7*(1+(2*x[:,0]-1)**2)/((90**2-(90*2*(1-x[:,1]))**2)**2+2.5**2*90**2)

def normalChristina(x):
    return 0.8* tf.exp((-0.5*((x[:,0]-0.5)* (50 *(x[:,0]-0.5) -  15* (x[:,1]-0.5)) + (-15*(x[:,0]-0.5) + 5*(x[:,1]-0.5))* (x[:,1]-0.5))))
def normalChristina_np(x1,x2):
    return 0.8* np.exp((-0.5*((x1-0.5)* (50 *(x1-0.5) -  15* (x2-0.5)) + (-15*(x1-0.5) + 5*(x2-0.5))* (x2-0.5))))

def camel(x):
    return 8* tf.exp((-0.5*(2)*(( (x[:,0]-0.25)* (25.0 *(x[:,0]-0.25) -  15* (x[:,1]-0.25)) + (-15*(x[:,0]-0.25) + 5*(x[:,1]-0.25))* (x[:,1]-0.25))+((x[:,0]-0.75)* (25 *(x[:,0]-0.75) -  15* (x[:,1]-0.75)) + (-15*(x[:,0]-0.75) + 5*(x[:,1]-0.75))* (x[:,1]-0.75)))))+1e-6
def camel_np(x0,x1):
    return 8* np.exp((-0.5*(2)*(( (x0-0.25)* (25.0 *(x0-0.25) -  15* (x1-0.25)) + (-15*(x0-0.25) + 5*(x1-0.25))* (x1-0.25))+((x0-0.75)* (25 *(x0-0.75) -  15* (x1-0.75)) + (-15*(x0-0.75) + 5*(x1-0.75))* (x1-0.75)))))+1e-6

def MultiGauss(x):
    return  tf.exp(-4.*((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2+4*(x[:,1]-0.5)*(x[:,0]-0.5)+(x[:,2]-0.75)**2 + (x[:,3]-0.5)**2+(x[:,4]-0.5)**2 -(x[:,4]-0.5)*(x[:,5]-0.5)+ (x[:,5]-0.5)**2-8*(x[:,3]-0.5)*(x[:,5]-0.5)))+1e-6
    
def MultiGauss_np(x):
    x0,x1,x2,x3,x4,x5 = x
    return np.exp(-4.*((x0-0.5)**2 + (x1-0.5)**2+4*(x1-0.5)*(x0-0.5)+(x2-0.75)**2 + (x3-0.5)**2+(x4-0.5)**2 + (x5-0.5)**2-(x4-0.5)*(x5-0.5)-8*(x3-0.5)*(x5-0.5)))+1e-6

def MultiGauss2(x):
    return  tf.exp(-4.*((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2+4*(x[:,1]-0.5)*(x[:,0]-0.5)+(x[:,2]-0.75)**2 + (x[:,3]-0.5)**2))+1e-6
    
def MultiGauss_np2(x):
    x0,x1,x2,x3 = x
    return np.exp(-4.*((x0-0.5)**2 + (x1-0.5)**2+4*(x1-0.5)*(x0-0.5)+(x2-0.75)**2 + (x3-0.5)**2))+1e-6

def Wall10(x):
    return 1e3*tf.exp(-2.*(x[:,0]+x[:,1]+x[:,2]+x[:,3]+x[:,4]+x[:,5]+x[:,6]+x[:,7]+x[:,8]+x[:,9]-1.)**2) +1e-6
def Wall10_np(x):
    x0,x1,x2,x3,x4,x5,x6,x7,x8,x9 = x
    return 1e3*np.exp(-2.*(x0+x1+x2+x3+x4+x5+x6+x7+x8+x9-1.)**2)+1e-6




def loss_fn(func):
    x = dist.sample(750)
    logq = dist.log_prob(x)
    p = func(x)
    q = dist.prob(x)
    p = p/tf.reduce_mean(p/q)
    return tf.reduce_mean(p/q*(tf.log(p)-logq))



# In[16]:


loss = loss_fn(MultiGauss)
optimizer = tf.train.AdamOptimizer(1e-4)
grads = optimizer.compute_gradients(loss)
opt_op = optimizer.apply_gradients(grads)
sess.run(tf.global_variables_initializer())


# In[ ]:


start_time = time.time()
np_losses = []
global_step = []
for epoch in range(3000):
    _, np_loss, np_grads = sess.run([opt_op, loss, grads])
    global_step.append(epoch)
    np_losses.append(np_loss)
    #print(np_x,np_logq)
    if(np_loss != np_loss):
        print(epoch, np_loss)
        break
    if (np_loss < 0.05):
        break
    if epoch % 50 == 0:
        print(epoch, np_loss)
print(epoch, np_loss)
start = 0
plt.plot(np_losses[start:])
plt.savefig("MultiGauss-6dim-loss-debug.pdf",bbox_inches='tight',dpi=150)
Time = time.time() - start_time
Hours = Time//3600
Minutes = (Time - 3600*Hours)//60
Seconds = (Time - 3600*Hours - 60*Minutes)
print("Done")
print("--- %(STD)s hours, %(MIN)s minutes and %(SEC).2f seconds ---" %{'STD':Hours,'MIN':Minutes, 'SEC':Seconds})

"""
nsamples = 500000
x = dist.sample(nsamples)
X = dist.log_prob(x)
q = tf.exp(X)
dsig = MultiGauss2(x)

xPts, logqPts, qPts, dsigPts = sess.run([x,X,q,dsig])
figure = corner.corner(xPts, labels=[r"$x_1$", r"$x_2$",r"$x_3$",r"$x_4$",r"$x_5$",r"$x_6$"],
                       #weights=dsigPts/qPts,
                       #weights=1./qPts,
                       show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig("MultiGauss-4dim-sample.pdf",bbox_inches='tight',dpi=150)
"""

start_time = time.time()
import vegas

ndims = 6
#integ = vegas.Integrator([[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.],[0.,1.]])
integ = vegas.Integrator(ndims*[[0.,1.]])
#result = integ(Wall10_np, nitn=10, neval=400200)
integ(MultiGauss_np, nitn=10, neval=200000)
result = integ(MultiGauss_np, nitn=5, neval=25000)
print("VEGAS")
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))
Time = time.time() - start_time
Hours = Time//3600
Minutes = (Time - 3600*Hours)//60
Seconds = (Time - 3600*Hours - 60*Minutes)
print("Done")
print("--- %(STD)s hours, %(MIN)s minutes and %(SEC).2f seconds ---" %{'STD':Hours,'MIN':Minutes, 'SEC':Seconds})


# In[27]:


start_time = time.time()
nsamples = 25000
nitn=5
xsec=0.
invvar=0.
for i in range(nitn):
    x = dist.sample(nsamples)
    q = dist.prob(x)
    dsig = MultiGauss(x)
    mean, var = sess.run(tf.nn.moments(dsig/q,axes=[0]))
    xsec+=mean*nsamples/var
    invvar+=(nsamples/var)

xsec=xsec/invvar
var=1./invvar
stddev=np.sqrt(var)
#print('xsec = %.4f, var = %.4f, stddev = %.4f' % (mean,var/nsamples,np.sqrt(var/nsamples)))
print('xsec = %e, var = %e, stddev = %e' % (xsec,var,stddev))

Time = time.time() - start_time
Hours = Time//3600
Minutes = (Time - 3600*Hours)//60
Seconds = (Time - 3600*Hours - 60*Minutes)
print("--- %(STD)s hours, %(MIN)s minutes and %(SEC).2f seconds ---" %{'STD':Hours,'MIN':Minutes, 'SEC':Seconds})

print("DONE FOR DEBUGGING PURPOSES!")
