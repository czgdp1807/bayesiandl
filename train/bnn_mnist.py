from models.bnn_mnist import BNN_Normal_Normal
from datasets.read_data import read_idx
import numpy as np
import tensorflow as tf
from datetime import datetime

def random_shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]

def take_subset(x, y, size, count):
    return x[count*size:count*size + size], \
           y[count*size:count*size + size]

def train(input_data_path, target_data_path, batch_size=1, dataset_size=1,
          epochs=1, cache_result=True, use_cache=False):
    curr_date_time = str(datetime.now()).replace(' ', '_').replace(':', '_').replace('-', '_').replace('.', '_')
    logs = open("./logs/"+curr_date_time, 'w')
    inputs = read_idx(input_data_path, True, True, True, dataset_size, False, cache_result, use_cache)
    targets = read_idx(target_data_path, False, False, True, dataset_size, True, cache_result, use_cache)
    logs.write(str((inputs.shape, targets.shape)) + '\n')
    size = targets.shape[0]
    model = BNN_Normal_Normal(input_shape=(batch_size, 784))
    logs.write("Number of batches per epoch: " + str(size//batch_size) + '\n')
    logs.write("Number of epochs: " + str(epochs) + '\n')
    logs.write("Initial Loss %s"%(model.get_loss(inputs, targets, 10)) + '\n')
    for epoch in range(epochs):
        inputs, targets = random_shuffle(inputs, targets)
        for batch in range(size//batch_size):
            input_sub, target_sub = take_subset(inputs, targets, batch_size, batch)
            model.learn(input_sub, target_sub, 1/(size//batch_size))
        logs.write("Loss at completion of epoch %s is %s"%(epoch, model.get_loss(inputs, targets, 10)) + '\n')
        logs.write("Cross entropy at completion of epoch %s is %s"%(epoch,model.get_loss(inputs, targets, 1, 1., True)) + '\n')
        if epoch%20 == 0:
            model.save_weights("./weights/mnist/" + curr_date_time)
