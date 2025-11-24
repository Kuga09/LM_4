import numpy as np
from keras import models, layers

models = models.Sequential([
    layers.Dense(3, input_shape=(5,), activation='relu')

])

weights = np.full((5.3),0.5) #все веса равны 0.5
biases = np.zeros(3)
models.layers[0].set_weights([weights,biases])

X = np.array([1,2,3,4,5],[0,1,0,1,0])
Y = models.predict(X)

print('Enters: ',X)
print('Outputs: ',Y)