import tensorflow as tf
import numpy as np

def data_generator(data, sq_len, n_features):
    inputs = tf.zeros((len(data), 0, n_features))

    for i in range(sq_len):
        inputs = tf.concat((inputs, np.array(data.shift(-i))[:,np.newaxis,:]), axis = 1)

    return inputs[:-sq_len+1,:,:-1], inputs[:-sq_len+1, -1, -1]

class Transactional_Extension(tf.keras.layers.Layer):
    def __init__(self, per_step: int, k: int):
        super(Transactional_Extension, self).__init__()
        self.per_step = per_step
        self.k = k

    def call(self, input):
        pad = tf.tile(input[:, 0:1], [1, self.per_step - 1, 1])
        fix_data = tf.concat((pad, input), axis = 1)
        data_split = tf.convert_to_tensor([fix_data[:, i : i + self.per_step] for i in range(input.shape[1])])
        data_cells = tf.transpose(tf.convert_to_tensor([data_split[i : i + self.k] for i in range(data_split.shape[0] - self.k + 1)]), perm=[2, 0, 1, 3, 4])
        return data_cells