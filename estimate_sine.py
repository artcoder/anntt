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


# Read in white wine data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


# Print info on white wine
#print(white.info())

# Print info on red wine
#print(red.info())



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



# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)



#corr = wines.corr()
#sns.heatmap(corr, 
#            xticklabels=corr.columns.values,
#            yticklabels=corr.columns.values)
#sns seems to use plt internally		
#plt.show()





# Specify the data 
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array 
y=np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#print("before")
#print( X_test)

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

#print("after")
#print( X_test)


# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer 
model.add(Dense(8, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))




model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)



y_pred = model.predict(X_test)



score = model.evaluate(X_test, y_test,verbose=1)

print(score)


confusion_matrix(y_test, y_pred.round())

precision_score(y_test, y_pred.round())

recall_score(y_test, y_pred.round())

f1_score(y_test,y_pred.round())

cohen_kappa_score(y_test, y_pred.round())

