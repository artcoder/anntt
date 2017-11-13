#based on https://www.datacamp.com/community/tutorials/deep-learning-python

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
# make input data
#x = np.arange(0.0, math.pi * 2.0, 0.01).reshape(-1, 1)
x = np.arange(0.0, 10, 0.01)
#print("x:", x[0:10])

np.random.shuffle(x)
#print("x:", x[0:10])


#make output data
y1 = list(map(lambda i: math.sin(i), x))
y = np.asarray(y1).reshape(-1,1)

x = x.reshape(-1, 1)
#print("x:", x[0:10])

#print("y:", y)
print("length of y:", len(y))



# Split the data up in train and test sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print("length of x_train:", len(x_train))

#print("before")
#print( x_test)

# Define the scaler 
#scaler = StandardScaler().fit( x )

# Scale the train set
#x_train = scaler.transform(x_train)

# Scale the test set
#x_test = scaler.transform(x_test)

#print("after")
#print( x_test)


# Initialize the constructor
model = Sequential()

# Add an the input layer and hidden layer
model.add(Dense(100, activation='tanh', input_dim=1))
#model.add(keras.layers.Dropout(0.1))

# Add a hidden layer 
model.add(Dense(100, activation='tanh'))
#model.add(keras.layers.Dropout(0.1))

# Add an output layer 
model.add(Dense(1, activation='tanh'))


# was 'binary_crossentropy'
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
                   
#? batch_size= len(x_train)
model.fit(x_train, y_train, epochs=100, batch_size= 1, verbose=0)


y_pred = model.predict(x_train)

#print("x:", x[0:10])
#print("y_pred:", y_pred[0:10])
#print("y:", y[0:10])


plt.plot(x_train, y_pred,'r.')
plt.plot(x, y,'g,')
plt.show()

score = model.evaluate(x_test, y_test,verbose=1)
print("\nScore:", score)


#confusion_matrix(y_test, y_pred)
#print("confusion matrix:", confusion_matrix)

#precision_score(y_test, y_pred.round())

#recall_score(y_test, y_pred.round())

#f1_score(y_test,y_pred.round())

#cohen_kappa_score(y_test, y_pred.round())

