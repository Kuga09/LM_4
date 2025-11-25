import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train,y_train), _ = mnist.load_data()
x_train = x_train[:200].reshape((200,28*28)).astype('float32')/255
y_train = to_categorical(y_train[:200])

model = models.Sequential([
    layers.Dense(128, input_shape=(784,), activation='relu'),
    layers.Dense(10, activation='softmax')   
])

weights_before, biases_before = model.layers[0].get_weights()

print(f"weights_before:\n{weights_before}\n")
print(f"biases_before:\n{biases_before}\n")

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(x_train, y_train, epochs=1, verbose=1) 

weights_after, biases_after = model.layers[0].get_weights()


print(f"weights_after):\n{weights_after}\n")
print(f"biases_after:\n{biases_after}\n")


weights_changed = np.array_equal(weights_before, weights_after)
print(f"Веса равны: {weights_changed}")