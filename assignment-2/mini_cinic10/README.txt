MINI CINIC-10 - based on cinic 10: https://github.com/BayesWatch/cinic-10

The dataset is composed of 100000 32x32 pixels RGB images, divided in 10 classes. They are divided into 80000 train samples, 10000 validation samples and 10000 test samples.

CONTENTS:
- train.npz/val.npz/test.npz: train/validation/test datasets.
    Each dataset is a numpy array with the following contents:
    - "xs": X matrix of shape (N, 3072). Each line represents one image flattened. The first 1024 elements are the red channel, the next 1024 elements are the green channel, the next 1024 are the blue channel.
    - "ys": Y vector of classes, of shape (N, ).
    - The ith image in X matrix is respective to the ith class in Y matrix.
    One method to read the data in python is by using the following code:
    import numpy
    data = numpy.read('train.npz')
    xs, ys = data['xs'], data['ys']

- classes_and_names.csv: Table mapping classes numbers to their names.

- images: Directory with ~1000 images (~1% random sample of the original images). Each image is named "<set>-<i>.png". "i" is the position of the image in the respective X matrix. Example: "train-34.png" is the image stored at the 34th line in the train.npz X matrix.
