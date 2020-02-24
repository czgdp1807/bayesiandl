from train.bnn_mnist import train
import sys
input_data_path = sys.argv[1]
output_data_path = sys.argv[2]
batch_size = int(sys.argv[3])
epochs = int(sys.argv[4])
cache_result = bool(int(sys.argv[5]))
use_cache = bool(int(sys.argv[6]))

train(input_data_path, output_data_path,
      batch_size, epochs, cache_result, use_cache)
