import tensorflow as tf
from models.bnn_mnist import BNN_Normal_Normal
from datasets.read_data import read_idx

def test(file_path, input_shape, input_data, output_data, cache_result, use_cache):

    model = BNN_Normal_Normal(input_shape=input_shape)
    model.load_weights(file_path)
    inputs = read_idx(input_data, True, True, True, 10000, False, cache_result, use_cache)
    targets = read_idx(output_data, False, False, True, 10000, True, cache_result, use_cache)
    preds, total_loss = model.get_loss(inputs, targets, 1, 1., True)
    return preds, targets, total_loss
