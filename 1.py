import numpy as np
from keras import models, layers

model = models.Sequential([
    layers.Dense(10, input_shape=(5,))
])

weights, biases = model.layers[0].get_weights()

print(weights.shape)
print(biases.shape)
print(weights[:3,:5]) #матрица весов
print(biases[:10]) #первые 10 сдвигов