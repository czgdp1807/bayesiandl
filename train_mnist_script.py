"""
System Arguments
================

input_data_path: str
      Path to the MNIST train images.
output_data_path: str
      Path to the MNIST train labels.
batch_size: int
      Size of one batch.
epochs: int
      Number of epochs to be used for training.
cache_result: bool
      Cache the data after pre-processing. Set 1
      to save processing time and computation power.
use_cache: bool
      Set 1 is cached pre-processed data is to be used.
weight_file: str
      The file where weights are to be saved.
"""
from train.bnn_mnist import train
import sys
input_data_path = sys.argv[1]
output_data_path = sys.argv[2]
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
cache_result = bool(int(sys.argv[5]))
use_cache = bool(int(sys.argv[6]))
dataset_size = int(sys.argv[7])

train(input_data_path, output_data_path, batch_size,
      dataset_size, epochs, cache_result, use_cache)
