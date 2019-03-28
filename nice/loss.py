import tensorflow as tf
import numpy as np

class Loss:
    def __init__(self, func):
        self.func = func
        self.max = -1

    def call(self, y_true, y_pred):
        loss = tf.reduce_mean(self.func(y_true)*tf.log(self.func(y_true))) \
              -tf.reduce_mean(self.func(y_true)*tf.reduce_sum(tf.log(y_pred),axis=-1))
        max_true = tf.reduce_max(self.func(y_true))
        if max_true > self.max:
            self.max = max_true

        return loss

    def __call__(self, y_true, y_pred):
        return self.call(y_true, y_pred)

if __name__ == '__main__':
    tf.enable_eager_execution()

    loss = Loss(lambda x: tf.square(x))

    vals = tf.random.uniform((5,))
    vals2 = tf.random.uniform((5,))
    with tf.GradientTape() as t:
        t.watch(vals)
        y = loss(vals2,vals)

    print(vals,vals2,y,tf.reduce_sum(t.gradient(y,vals)))

    vals = tf.random.uniform((5,))
    vals2 = vals
    with tf.GradientTape() as t:
        t.watch(vals)
        y = loss(vals2,vals)

    print(vals,vals2,y,tf.reduce_sum(t.gradient(y,vals)))
