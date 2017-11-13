# Python 3
# Based on code from
# https://www.datacamp.com/community/tutorials/deep-learning-python

# Import pandas 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import math

# using the form: y = f(x)
# x is the input
# y is the output

# make input data
x = np.arange(0.0, 10, 0.01)
#print("x:", x[0:10])

#np.random.shuffle(x)
#print("x:", x[0:10])


# make output data
y1 = list(map(lambda i: math.sin(i), x))
y = np.asarray(y1).reshape(-1,1)

x = x.reshape(-1, 1)
#print("x:", x[0:10])

#print("y:", y)
print("length of y:", len(y))



# Split the data up in train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y)
print("length of x_train:", len(x_train))

# Initialize the constructor
model = Sequential()

# Add an the input layer and first layer in one step
# tanh is used since it has positive and negative output like the sine function
model.add(Dense(100, activation='tanh', input_dim=1))
#model.add(keras.layers.Dropout(0.1))

# Add a hidden layer 
model.add(Dense(100, activation='tanh'))
#model.add(keras.layers.Dropout(0.1))

# Add the output layer 
model.add(Dense(1, activation='tanh'))


model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
                   
history = model.fit(x_train, y_train, epochs=10000, batch_size= 100, verbose=0)
#plt.plot(history.history['mean_squared_error'])

y_pred = model.predict(x_train)
plt.draw()

#print("x:", x[0:10])
#print("y_pred:", y_pred[0:10])
#print("y:", y[0:10])

# prediction in red
plt.plot(x_train, y_pred,'r.')
# truth in green
plt.plot(x, y,'g,')
plt.draw()

score = model.evaluate(x_test, y_test, batch_size= 1, verbose=1)
print("\nScore:", score)


plt.show() # A blocking call to keep the window open
