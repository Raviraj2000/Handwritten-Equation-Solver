import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv", index_col = False)
labels = df_train[['784']]

df_train.drop(df_train.columns[[784]], axis = 1, inplace = True)
df_train.head()


import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential 
from keras.utils import np_utils 


labels = np.array(labels)

cat = np_utils.to_categorical(labels, num_classes = 13)

print(cat[0])

l = []

for i in range(156617):
    l.append(np.array(df_train[i:i+1]).reshape(28,28,1))
    
model = Sequential()

model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(15, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(13, activation = "softmax"))

model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics = ['accuracy'])


model.summary()

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96)

model.fit(np.array(l), cat, epochs=10, batch_size=200,shuffle=True,verbose=1)

model_json = model.to_json()

with open("handwritten_equation_solver.json", "w") as json_file:
    json_file.write(model_json)


model.save_weights("handwritten_equation_solver.h5")

