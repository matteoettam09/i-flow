import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from .layers import *

def build_relu(input_dim,hidden_dim):
    in_vals = layers.Input(shape=(input_dim,))
    hidden = layers.Dense(hidden_dim,activation='relu')(in_vals)
    hidden = layers.Dense(hidden_dim,activation='relu')(hidden)
    hidden = layers.Dense(hidden_dim,activation='relu')(hidden)
    output = layers.Dense(input_dim)(hidden)

    model = models.Model(inputs=in_vals, outputs=output)
    model.summary()
    return model

def build_NICE(input_dim,hidden_dim):
    half_dim = int(input_dim/2)
    in_vals = layers.Input(shape=(input_dim,))
    hidden = AdditiveCouplingLayer(input_dim, 'odd', build_relu(half_dim,hidden_dim))(in_vals)
    hidden = AdditiveCouplingLayer(input_dim, 'even', build_relu(half_dim,hidden_dim))(hidden)
    hidden = AdditiveCouplingLayer(input_dim, 'odd', build_relu(half_dim,hidden_dim))(hidden)
    output = AdditiveCouplingLayer(input_dim, 'even', build_relu(half_dim,hidden_dim))(hidden)

    return models.Model(inputs=in_vals, outputs=output)

if __name__ == '__main__':
    model = build_NICE(4,4)
    model.summary()
