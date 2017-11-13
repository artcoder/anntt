#based on https://www.datacamp.com/community/tutorials/deep-learning-python

# Import pandas 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import math

# using the variables in the form: y = f(x)
# make input data
x = np.arange(0.0, math.pi * 2.0, 0.01).reshape(-1, 1)

#make output data
y1 = list(map(lambda i: math.sin(i), x))
y = np.asarray(y1).reshape(-1,1)

#print("y:", y)
print("length of y:", len(y))

#fig, ax = plt.subplots(1, 2)

#ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
#ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

#fig.subplots_adjust(left=0.125, right=.9, bottom=0.1, top=0.9, hspace=0.2, wspace=0.5)
#ax[0].set_ylim([0, 1000])
#ax[0].set_xlabel("Alcohol in % Vol")
#ax[0].set_ylabel("Frequency")
#ax[1].set_xlabel("Alcohol in % Vol")
#ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
#fig.suptitle("Distribution of Alcohol in % Vol")

#plt.show()



# Split the data up in train and test sets
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

print("length of x_train:", len(x_train))

#print("before")
#print( x_test)

# Define the scaler 
scaler = StandardScaler().fit( x )

# Scale the train set
#x_train = scaler.transform(x_train)

# Scale the test set
#x_test = scaler.transform(x_test)

#print("after")
#print( x_test)


# Initialize the constructor
model = Sequential()

# Add an input layer and hiddel layer
model.add(Dense(100, activation='relu', input_dim=1))

# Add a hidden layer 
#model.add(Dense(100, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='tanh'))




model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(x_train, y_train, epochs=1000, batch_size=400, verbose=1)


y_pred = model.predict(x_test)

print("x_test:", x_test[0:10])
print("y_pred:", y_pred[0:10])
print("y_test:", y_test[0:10])




score = model.evaluate(x_test, y_test,verbose=1)
print("\nScore:", score)


#confusion_matrix(y_test, y_pred)
#print("confusion matrix:", confusion_matrix)

#precision_score(y_test, y_pred.round())

#recall_score(y_test, y_pred.round())

#f1_score(y_test,y_pred.round())

#cohen_kappa_score(y_test, y_pred.round())

