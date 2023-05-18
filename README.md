import tensorflow as tfy 
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten, Conv2D, MaxPooling2D
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
plt.imshow(x_train[0])
plt.show()
plt.imshow(x_train[0],cmap=plt.cm.binary)
print(x_train[0])
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
plt.imshow(x_train[0],cmap=plt.cm.binary)
print(x_train[0])
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
model = Sequential()

# layer 1
model.add(Conv2D(64,(3,3),input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 2
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 3
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# fully connected layer 1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

# fully connected layer 2
model.add(Dense(32))
model.add(Activation("relu"))

# fully connected layer 3
model.add(Dense(10))
model.add(Activation('softmax'))
model.summary()
model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,validation_split=0.3)
test_loss, test_acc = model.evaluate(x_testr,y_test)
print("test loss on 10000 test samples : ",test_loss)
print("validation Accuracy on 10000 test samples : ",test_acc)
predictions = model.predict([x_testr])
print(predictions)
print(np.argmax(predictions[99]))
plt.imshow(x_test[99])
img = cv2.imread("/content/five.png")
plt.imshow(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray.shape
resized = cv2.resize(gray,(28,28),interpolation = cv2.INTER_AREA)
newimg = tf.keras.utils.normalize(resized,axis=1)
newimg = np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)
newimg.shape
print(np.argmax(model.predict(newimg)))
