from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D , Dense , Dropout , Flatten , MaxPool2D
from keras.optimizers import SGD

#Loading the Data from Server
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

#Checking how the digits look by using matplotlib
# random_number = np.random.randint(0, len(x_train))
# plt.imshow(x_train[random_number] , cmap=plt.get_cmap('gray'))
# plt.show()

#Get the number of Rows and Columns
rows = x_train[0].shape[0]
column = x_train[1].shape[0]

#Reshape to match the Input
x_train = x_train.reshape(x_train.shape[0] , rows , column , 1)
x_test = x_test.reshape(x_test.shape[0], rows, column, 1)
imageshape = ( rows, column , 1 )

#Change the DataType to Float32 for normalization, Because it is Int at first

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalise the data , Bring it between 0 and 1

x_train /= 255
x_test /= 255

#Chnage the Label Data to One Hot encoding format for Keras to Support it 

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

#Start Creating Model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3) , activation='relu' , input_shape=imageshape ))
model.add(Conv2D( 64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes , activation='softmax'))

#Adding the loss measure , SGD and specifying the learning Rate
model.compile(loss='categorical_crossentropy' , optimizer=SGD(0.01) , metrics=['accuracy'])
print(model.summary())

#Train the model to get the accuracy
batch_size = 32
epochs = 2

#Store the History to Evaluate it in 

history = model.fit(x_train , y_train , batch_size=batch_size, epochs=epochs , verbose=1 ,validation_data=( x_test , y_test ))
performance = model.evaluate(x_test, y_test, verbose=0)

#Checking the performance of Model using the Graphs

data = history.history

training_loss = data['loss']
validation_loss = data['val_loss']
epochs = range(1, len(training_loss) + 1)

line1 = plt.plot(epochs , validation_loss , label='Validation/TestLoss')
line2 = plt.plot(epochs, training_loss, label='TrainingLoss')


plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.grid(True)
plt.legend()
plt.show()

