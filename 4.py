import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

(x_train,y_train), _ = mnist.load_data()
x_train = x_train[:200].reshape((200,28*28)).astype('float32')/255
y_train = to_categorical(y_train[:200])

models = models.Sequential([
        layers.Dense(64, kernel_initializer=init, input_shape=(100,)),
        layers.Dense(10, activation='softmax')
    ])
model.compile(optimizer='adam',loss='categorial_crossentropy')
w_before,b_before = model.layer[0].get_weights()
model.fit(x_train,y_train,epochs=1,batch_size=32,verbose=1)
w_after,b_after = model.layers[0].get_weights()
delta_w = w_after-w_before

print('макс изменение веса:',np.max(np.abs(delta_w)))
print('среднее изменение веса:',np.mean(np.abs(delta_w)))