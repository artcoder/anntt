# for Python 3.6
#
# Estimate Sine by Shape
#
# Inspired by code from a tutorial at:
# https://www.datacamp.com/community/tutorials/deep-learning-python

import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

EPOCHS = 10000
BATCH_SIZE = 100

# Using the form: y = f(x)
# x is the input
# y is the output

# Make the input data
raw_data = [ math.sin(i) for i in np.arange(0.0, 10, 0.01) ]
time = np.arange(0.0, 10, 0.01)

width = 10
x = []
y = []
for i in range(0, len(raw_data)-width-1):
    #input data
    # the item x[i] is "width" consecutive outputs of sine(i)
    x.append(raw_data[i:i+width])

    #output data
    # y is the value of sine(i) following the last one in x[i]
    y.append([raw_data[i+width]])

#print("x:", x[0:5])
#print("y:", y[0:5])
#print("time:", time[0:5])

# np.random.shuffle(x)
# print("x:", x[0:10])

# print("y:", y)
#print("length of x:", len(x))
#print("length of y:", len(y))

# Split the data up in training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y)
# print("length of x_train:", len(x_train))

# start a Keras model
model = Sequential()

# Add an the input layer and first layer in one step
# tanh is used since it has positive and negative output like the sine function
model.add(Dense(100, activation='tanh', input_dim=10))
# model.add(keras.layers.Dropout(0.1))

# Add a hidden layer
model.add(Dense(100, activation='tanh'))
# model.add(keras.layers.Dropout(0.1))

# Add the output layer
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_absolute_error', optimizer='adam',
              metrics=['mean_squared_error'])

history = model.fit(x_train, y_train, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, verbose=0)
# plt.plot(history.history['mean_squared_error'])
# plt.draw()

#y_pred = model.predict(x_train)

score = model.evaluate(x_test, y_test, batch_size=1, verbose=1)
print("\nError score:", score)

y_pred = model.predict(x)


plt.title('Neural Network Estimation by Shape')

plt.plot(time[0:len(y_pred)], y_pred, label='Prediction',
         color='red', marker='.', linestyle='None')
plt.plot(time[0:len(y_pred)], y[0:len(y_pred)], label='Target', color='green', marker=',', linestyle='None')
plt.legend()
plt.text(5.3, -.98, 'Epochs={}, Batch size={}'.format(EPOCHS, BATCH_SIZE))
plt.draw()

# Make a blocking call to keep the pyplot window open
plt.show()
