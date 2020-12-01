import numpy as np
import math
import random
from keras.models import Sequential
from keras.layers import Dense

def generate_points():
	array_x = []
	array_y = []
	for i in range(100):
		x = random.uniform(0,1)
		y = random.uniform(0,1)
		array_x.append( x )
		array_y.append( y )
	return ( array_x, array_y )

def concept( x, y ):
	circ = ( ( x - 0.5 ) * ( x - 0.5 ) ) + ( ( y - 0.6 ) * ( y - 0.6 ) )
	r_s = 0.4 * 0.4
	if( circ < r_s ):
		return True
	else: return False

train_points = xt,yt = generate_points()
train_labels = []
test_points = x_t,y_t  = generate_points()
test_labels = []


for i,j in zip(xt,yt):
	train_labels.append( concept(i,j) )
	#print( concept( i,j ) )

for i,j in zip(x_t,y_t):
	test_labels.append( concept(i,j) )
	#print( concept( i ) )


model = Sequential()
model.add(Dense(2, input_dim=2, activation='sigmoid',kernel_initializer='random_normal'))
model.add(Dense(5, activation='sigmoid', kernel_initializer='random_normal'))
model.add(Dense(5, activation='sigmoid', kernel_initializer='random_normal'))
model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())
print(model.layers[3].get_weights())

train_points = np.array(train_points)
train_labels = np.array(train_labels)

print(train_points.shape)

model.fit(np.transpose(train_points), train_labels, epochs=50, batch_size = 5)

_,accuracy = model.evaluate(np.transpose(test_points), test_labels)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict_classes(np.transpose(test_points))
# summarize the first 5 cases

model.summary()
print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())
print(model.layers[3].get_weights())
