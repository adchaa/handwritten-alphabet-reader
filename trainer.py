import numpy as np
from open_file import open_file
import tensorflow.keras as ker

file = open_file("A_Z Handwritten Data.csv")
print("loading data...")
(x_train,y_train),(x_test,y_test) = file.load_data()


x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test, -1)

# model
print("model train")
input()
model = ker.models.Sequential()

model.add(ker.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation=ker.activations.relu))
model.add(ker.layers.Conv2D(32,(3,3),activation=ker.activations.relu))
model.add(ker.layers.MaxPool2D((2,2)))
model.add(ker.layers.Conv2D(64,(3,3),activation=ker.activations.relu))
model.add(ker.layers.Flatten())
model.add(ker.layers.Dense(26,activation=ker.activations.softmax))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=2)
print(model.evaluate(x_test,y_test))
model.save("letter_regination.h5")
