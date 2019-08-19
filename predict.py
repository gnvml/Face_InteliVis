from utils.annModel import Model
import numpy as np

model = Model()
x = np.load("train_data.npy")
print(x.shape)
name = model.predict(x)
print(name)