import numpy as np
from keras import models, layers

model = models.Sequential([
    layers.Dense(3, input_shape=(5,), activation='relu', kernel_initializer='zeros')
])

weights, biases = model.layers[0].get_weights()

X = np.array([[1,2,3,4,5],[0,1,0,1,0]])
Y = model.predict(X)

print('Enters: ',X)
print('Outputs: ',Y)