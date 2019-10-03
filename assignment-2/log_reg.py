import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from nene import *

PATH_TO_DATA = os.path.join("mini_cinic10")

IMG_WIDTH = IMG_HEIGHT = 32
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3) # RGB images
IMG_FLAT_SHAPE = (IMG_HEIGHT*IMG_WIDTH*3, )

# classes_and_names.csv
CLASS_NAME = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
CLASS_COUNT = len(CLASS_NAME)

train_data = np.load(os.path.join(PATH_TO_DATA, "train.npz"))
val_data   = np.load(os.path.join(PATH_TO_DATA, "val.npz"))
# test_data  = np.load(os.path.join(PATH_TO_DATA, "test.npz")) # assume this doesn't exist

xs, ys = train_data['xs'], train_data['ys']
xs_val, ys_val = val_data['xs'], val_data['ys']

# NOTE that we must use stats from train data to normalize the val and test sets aswell
mean, std = xs.mean(), xs.std()
X = (xs - mean) / std
X_val = (xs_val - mean) / std

def onehot_encode(ys):
    n_examples, *_ = ys.shape
    onehot = np.zeros(shape=(n_examples, CLASS_COUNT))
    onehot[np.arange(n_examples), ys] = 1
    return onehot

Y = onehot_encode(ys)
Y_val = onehot_encode(ys_val)

###############################################################################

"""
nn_log_reg = NN(
    cost_function=CrossEntropy(),
    optimizer=GradientDescent(learning_rate=1e-4),
    weight_initialization='xavier',
    layers=[
        Layer(3072, None), Layer(10, SoftMax())
    ])

nn_log_reg.train(
    X[:2*1250], Y[:2*1250],
    X_val, Y_val,
    n_epochs=2,
    batch_size=1250//2
) # X, Y, X_val, Y_val, n_epochs, batch_size, verbose=True

print(nn_log_reg.history)
"""

nn2_log_reg = NN(
    cost_function=SoftmaxCrossEntropy(),
    optimizer=GradientDescent(learning_rate=1e-4),
    weight_initialization='xavier',
    layers=[
        Layer(3072, None), Layer(10, Linear())
    ])

start = time()
print("Starting to train...")
nn2_log_reg.train(
    X[:4*2048], Y[:4*2048],
    X_val, Y_val,
    n_epochs=2,
    batch_size=1024
) # X, Y, X_val, Y_val, n_epochs, batch_size, verbose=True
end = time()

print(f"\nDone. Training took {end - start}s")

print(f"\nTrain history:")
for k, v in nn2_log_reg.history.items():
    print(f"{k}: {v}")

