import struct

import numpy as np

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def read_idx(filename, flatten=True, normalize=True, show_logs=True, num_data=1000000, to_one_hot=False):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        ret_val = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        num_data = min(ret_val.shape[0], num_data)
        normalize_val = []
        if normalize:
            for i in range(num_data):
                mat = [[0 for _i in range(ret_val.shape[1])]
                        for _j in range(ret_val.shape[2])]
                if show_logs:
                    print("Normalized %s-th data"%(i+1))
                for j in range(ret_val.shape[1]):
                    for k in range(ret_val.shape[2]):
                        mat[j][k] = ret_val[i][j][k]/255.
                normalize_val.append(mat)
                del mat
            ret_val = np.asarray(normalize_val)
        del normalize_val
        flatten_val = []
        if flatten:
            for i in range(num_data):
                if show_logs:
                    print("Flattened %s-th data"%(i+1))
                flatten_val.append(ret_val[i].flatten('C'))
            ret_val = np.asarray(flatten_val)
        del flatten_val
        if to_one_hot:
            ret_val = one_hot(ret_val, 10)
        return ret_val
