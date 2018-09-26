from keras import models
from keras import layers
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class Neural_network:
    def __init__(self):
        pass

    def fit_predict(self,train_data,train_labels,test_data):
        #CONVERT LABELS TO ONE HOT AS IT USES SOFTMAX LAYER AT END
        one_hot_obj = OneHotEncoder()
        train_labels_encoded = one_hot_obj.fit_transform(np.array(train_labels).reshape(-1,1)) #train_labels are list, converting them to shape(-1,1)
        model = models.Sequential()
        # Input - Layer
        model.add(layers.Dense(80, activation="relu"))
        # Hidden - Layers
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(60, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(50, activation="relu"))
        # Output- Layer
        model.add(layers.Dense(4, activation="softmax"))
        # compiling the model
        model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        model.fit(train_data.todense(), train_labels_encoded.todense(),epochs=20,batch_size=500)
        predictions = model.predict_classes(test_data.todense())
        return predictions