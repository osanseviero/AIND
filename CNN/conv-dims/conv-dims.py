from math import ceil
from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding="same",
	activation="relu", input_shape=(128, 128, 3)))
model.summary()

# Parameters = K * Weights + K = K * F * F * D_in + K
number_parameters = 32 * 3 * 3 * 3 + 32

# Shape of convolutional layer

# Depth = K
depth = 32

# Since it is a same padding: height=ceil(H_in/S)
height = ceil(float(128)/float(2))
width = ceil(float(128)/float(2))

print("Number of parameters: ", number_parameters)
print("Shape of Convolutional layer (%d,%d,%d)" % (height, width, depth))