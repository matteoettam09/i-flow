import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

_get_even = lambda xs: xs[:,0::2]
_get_odd = lambda xs: xs[:,1::2]

def _interleave(first, second, order):
    cols = []
    if order == 'even':
        for k in range(second.shape[1]):
            cols.append(first[:,k])
            cols.append(second[:,k])
        if first.shape[1] > second.shape[1]:
            cols.append(first[:,-1])
    else:
        for k in range(first.shape[1]):
            cols.append(second[:,k])
            cols.append(first[:,k])
        if second.shape[1] > first.shape[1]:
            cols.append(second[:,-1])

    return tf.stack(cols, axis=1)

class _BaseCouplingLayer(layers.Layer):
    def __init__(self, dim, partition, nonlinearity):
        super(_BaseCouplingLayer, self).__init__()
        self.dim = dim
        self.partition = partition 
        self.nonlinearity = nonlinearity

    def build(self, dim):
        self.dim = dim
        if self.partition == 'even':
            self._first = _get_even
            self._second = _get_odd
        else:
            self._first = _get_odd
            self._second = _get_even

    def call(self, x):
        return _interleave(
                self._first(x),
                self.coupling_law(self._second(x), self.nonlinearity(self._first(x))),
                self.partition
                )

    def inverse(self, y):
        return _interleave(
                self._first(y),
                self.anticoupling_law(self._second(y), self.nonlinearity(self._first(y))),
                self.partition
                )

    def coupling_law(self, a, b):
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer")

    def anticoupling_law(self, a, b):
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer")

    def jacobian(self,a,b):
        raise NotImplementedError("[_BaseCouplingLayer] Don't call abstract base layer")

class AdditiveCouplingLayer(_BaseCouplingLayer):
    def coupling_law(self,a,b):
        return (a + b)
    def anticoupling_law(self,a,b):
        return (a - b)
    def jacobian(self,a,b):
        return 1.0

class MultiplicativeCouplingLayer(_BaseCouplingLayer):
    def coupling_law(self,a,b):
        return tf.multiply(a,b)
    def anticoupling_law(self,a,b):
        return tf.multiply(a,tf.reciprocal(b))
    def jacobian(self,a,b):
        return tf.reduce_prod(tf.reciprocal(b))

class AffineCouplingLayer(_BaseCouplingLayer):
    def coupling_law(self,a,b):
        return tf.multiply(a, self._first(b)) + self._second(b)
    def anticoupling_law(self,a,b):
        return tf.multiply(a-self._second(b),tf.reciprocal(self._first(b)))

class PiecewiseLinear(_BaseCouplingLayer):
    def __init__(self, dim, partition, nonlinearity, nbins):
        super(PiecewiseLinear, self).__init__(dim,partition,nonlinearity)
        self.nbins = nbins
        self.width = 1.0/self.nbins

    def coupling_law(self,b,Q):
        alpha, loc = self.get_alpha_bin(b)
        result = np.zeros_like(loc)
        for i,_ in enumerate(loc):
            result[i] = alpha[i]*Q[i,int(loc[i])]
            result[i] += tf.reduce_sum(Q[i,:int(loc[i])])

        return result

    def jacobian(self,b,Q):
        loc = self.find_bin(b)
        result = 1
        for i,_ in enumerate(loc):
            result *= Q[i,int(loc[i])]
        return result/self.width

    def find_bin(self,x):
        loc = np.zeros(int(self.dim/2))
        binhigh = self.width
        for i in range(int(self.dim/2)):
            while x[i] > binhigh:
                binhigh += self.width
                loc[i] += 1
        return loc

    def get_alpha_bin(self,x):
        loc = self.find_bin(x)
        binlow = self.width*loc
        alpha = x-binlow
        return alpha, loc

class PiecewiseQuadratic(_BaseCouplingLayer):
    def __init__(self, dim, partition, nonlinearity,nbins):
        super(PiecewiseLinear, self).__init__(dim,partition,nonlinearity)
        self.nbins = nbins

    def coupling_law(self,b,Q):
        alpha, loc = self.get_alpha_bin(b)
        result = np.zeros_like(loc)
        for i,_ in enumerate(loc):
            result[i] = tf.square(alpha[i])/2.0*(V[i,int(loc[i])+1]-V[i,int(loc[i])]) \
                      + alpha*V[i,int(loc[i])]
            result[i] += tf.reduce_sum((V[i,:int(loc[i])]+V[i,:int(loc[i])+1)/2.0*W[i,:int(loc[i])])

        return result


if __name__ == '__main__':
    tf.enable_eager_execution()

    layerAdd = AdditiveCouplingLayer(2,'even',lambda x: x)
    layerAdd.build(2)
    print(layerAdd.coupling_law(1.0,1.0))
    print(layerAdd.anticoupling_law(2.0,1.0))
    print(layerAdd.jacobian(1.0,1.0))

    layerMul = MultiplicativeCouplingLayer(2,'even',lambda x: x)
    layerMul.build(2)
    print(layerMul.coupling_law(2.0,3.0))
    print(layerMul.anticoupling_law(6.0,2.0))
    print(layerMul.jacobian(2.0,3.0))

    layerPiecewiseLinear = PiecewiseLinear(4,'even',lambda x: ((x,x,x,x),(x,x,x,x)),4)
    layerPiecewiseLinear.build(4)
    f = lambda x: tf.constant([[x,x,x,x],[x,x,x,x]])
    print(layerPiecewiseLinear.coupling_law([0.2,0.5],f(0.2)))
    print(layerPiecewiseLinear.jacobian([0.2,0.5],f(0.2)))
