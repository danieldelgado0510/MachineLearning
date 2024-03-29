#VAE implemented using tensorflow
#Daniel Delgado
#CAP6610

from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

#print( train_images.shape )
#print( test_images.shape )

def preprocess_images(images):
	images = images.reshape((images.shape[0], 28, 28, 1))/255.
	return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
#batch_size = 256
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

#print( train_dataset.shape )
#print( test_dataset.shape )

#VAE model as a class
class CVAE(tf.keras.Model):

	def __init__(self, latent_dim):
		super(CVAE, self).__init__()
		self.latent_dim = latent_dim
		self.encoder = tf.keras.Sequential(
			[
				#layer 1
				tf.keras.layers.InputLayer(input_shape=(28,28,1)),
				#layer 2
				tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'),
				#layer 3
				tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'),
				#layer 4
				tf.keras.layers.Flatten(),
				#layer 5
				tf.keras.layers.Dense(latent_dim + latent_dim),
			]
		)

		self.decoder = tf.keras.Sequential(
			[
				#layer 1
				tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
				#layer 2
				tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
				#layer 3
				tf.keras.layers.Reshape(target_shape=(7,7,32)),
				#layer 4
				tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
				tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
				tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
			]
		)

	@tf.function
	def sample(self, eps=None):
		if eps is None:
			eps = tf.random.normal(shape=(100, self.latent_dim))
		return self.decode(eps, apply_sigmoid=True)

	#calls on model encoder
	def encode(self, x):
		mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
		return mean, logvar

	def reparameterize(self, mean, logvar):
		eps = tf.random.normal(shape=mean.shape)
		return eps * tf.exp(logvar * .5) + mean

	#calls on model decoder
	def decode(self, z, apply_sigmoid=False):
		logits = self.decoder(z)
		if apply_sigmoid:
			probs = tf.sigmoid(logits)
			return probs
		return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

#Loss Function Computation, based on maximizing the 
#evidence lower bound on the marginal log-likelihood
def log_normal_pdf(sample, mean, logvar, raxis=1):
	log2pi = tf.math.log(2. * np.pi)
	return tf.reduce_sum(-0.5 * ((sample-mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis = raxis)

def compute_loss(model, x):
	mean, logvar = model.encode(x)
	z = model.reparameterize(mean, logvar)
	x_logit = model.decode(z)
	cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
	logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2,3])
	logpz = log_normal_pdf(z,0.,0.)
	logqz_x = log_normal_pdf(z, mean, logvar)
	return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
	with tf.GradientTape() as tape:
		loss = compute_loss(model,x)
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 50
#epochs = 10
latent_dim = 2
num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
	mean, logvar = model.encode(test_sample)
	z = model.reparameterize(mean, logvar)
	predictions = model.sample(z)
	fig = plt.figure(figsize=(4,4))

	for i in range(predictions.shape[0]):
		plt.subplot(4, 4, i+1)
		plt.imshow(predictions[i, :, :, 0])
		plt.axis('off')

	plt.savefig('images/images_at_epoch{:04}.png'.format(epoch), cmap='gray')

assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
	test_sample = test_batch[0:num_examples_to_generate, :, :, :]

generate_and_save_images(model, 0, test_sample)

#Run and train model
for epoch in range(1, epochs + 1):
	start_time = time.time()
	for train_x in train_dataset:
		train_step(model, train_x, optimizer)
	end_time = time.time()
	loss = tf.keras.metrics.Mean()
	for test_x in test_dataset:
		loss(compute_loss(model, test_x))
	elbo = -loss.result()
	print( epoch, elbo )
	display.clear_output(wait=False)
	print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))
	generate_and_save_images(model, epoch, test_sample)

