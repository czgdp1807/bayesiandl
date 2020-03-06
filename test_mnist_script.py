from test.bnn_mnist import test
import sys
input_data = sys.argv[1]
output_data = sys.argv[2]
batch_size = int(sys.argv[3])
input_len = int(sys.argv[4])
weight_file = sys.argv[5]
cache_result = bool(int(sys.argv[6]))
use_cache = bool(int(sys.argv[7]))


outputs, targets, loss = test(weight_file, (batch_size, input_len), input_data, output_data, cache_result, use_cache)
print("Cross Entropy Loss: ", loss)
for i in range(outputs[0].shape[0]):
    print("Output %s "%(i), outputs[0][i])
    print("Target %s "%(i), targets[i])
