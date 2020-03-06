from models.bnn_mnist import BNN_Normal_Normal
from datasets.read_data import read_idx
import numpy as np
import tensorflow as tf
from datetime import date

def random_shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]

def take_subset(x, y, size, count):
    return x[count*size:count*size + size], \
           y[count*size:count*size + size]

def train(input_data_path, target_data_path, weight_file, batch_size=1, dataset_size=1,
          epochs=1, cache_result=True, use_cache=False):
    inputs = read_idx(input_data_path, True, True, True, dataset_size, False, cache_result, use_cache)
    targets = read_idx(target_data_path, False, False, True, dataset_size, True, cache_result, use_cache)
    print(inputs.shape, targets.shape)
    size = targets.shape[0]
    model = BNN_Normal_Normal(input_shape=(batch_size, 784))
    print("Number of batches per epoch:", size//batch_size)
    print("Number of epochs:", epochs)
    print("Initial Loss %s"%(model.get_loss(inputs, targets, 10)))
    for epoch in range(epochs):
        inputs, targets = random_shuffle(inputs, targets)
        for batch in range(size//batch_size):
            input_sub, target_sub = take_subset(inputs, targets, batch_size, batch)
            model.learn(input_sub, target_sub, 1/(size//batch_size))
        print("Loss at completion of epoch %s is %s"%(epoch, model.get_loss(inputs, targets, 10)))
        print("Cross entropy at completion of epoch %s is %s"%(epoch,model.get_loss(inputs, targets, 1, 1., True)))
        if epoch%20 == 0:
            model.save_weights("./weights/mnist/" +
             date.today().strftime("%d_%m_%Y"))
