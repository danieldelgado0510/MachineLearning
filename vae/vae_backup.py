import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

encoding_dim = 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
	 epochs=50,
	 batch_size=256,
	 shuffle=True,
	 validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 25
plt.figure(figsize=(20,4))

for i in range(n):

	ax = plt.subplot(2,n,i+1)
	plt.imshow(x_test[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)
	
	ax = plt.subplot(2,n,i+1+n)
	plt.imshow(decoded_imgs[i].reshape(28,28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_xaxis().set_visible(False)

plt.savefig('vae.png')
"""
for i in range(n):
	plt.subplot(n,n,1+i)
	plt.axis('off')
	plt.imshow(decoded_imgs[i].reshape(28,28), cmap='gray_r')
	plt.savefig('vae.png')
"""
output = Dense(10, activation='softmax')(encoded)
autoencoder = Model(input_img,output)
autoencoder.compile(optimizer='adadelta',loss='categorical_crossentropy',metrics=['accuracy'])
autoencoder.fit(x_train,y_train,epochs=5,batch_size=32)
score = autoencoder.evaluate(x_test,y_test,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
