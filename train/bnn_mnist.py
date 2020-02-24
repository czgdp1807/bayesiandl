from models.bnn_mnist import BNN_Normal_Normal
from datasets.read_data import read_idx
import numpy as np
import tensorflow as tf

def random_shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]

def take_subset(x, y, size, count):
    return x[count*size:count*size + size], \
           y[count*size:count*size + size]

def train(input_data_path, target_data_path, batch_size=1, epochs=1,
          cache_result=True, use_cache=False):
    inputs = read_idx(input_data_path, True, True, True, 100, False, cache_result, use_cache)
    targets = read_idx(target_data_path, False, False, True, 100, True, cache_result, use_cache)
    print(inputs.shape, targets.shape)
    size = targets.shape[0]
    model = BNN_Normal_Normal(input_shape=(batch_size, 784))
    for epoch in range(epochs):
        inputs, targets = random_shuffle(inputs, targets)
        for batch in range(size//batch_size):
            tf.compat.v1.logging.info(
                "Epoch %s, Batch %s ..."%(epoch, batch))
            input_sub, target_sub = take_subset(inputs, targets, batch_size, batch)
            model.learn(input_sub, target_sub, 1/batch_size)
