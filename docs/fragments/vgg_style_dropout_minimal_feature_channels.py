from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D

# input tensor for a 3-channel 64x64 image
inputs = Input(shape=(64, 64, 3))

# one block of convolutional layers
x = Convolution2D(8, 3, activation='relu', padding='same')(inputs)
x = Dropout(0.5)(x)
# x = Convolution2D(64, 3, activation='relu', padding='same')(x)
# x = Convolution2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# one more block
x = Convolution2D(16, 3, activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
# x = Convolution2D(128, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# one more block
x = Convolution2D(32, 3, activation='relu', padding='same')(x)
x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)
x = Dense(32, activation='sigmoid')(x)

# softmax activation, 6 categories
predictions = Dense(6, activation='softmax')(x)