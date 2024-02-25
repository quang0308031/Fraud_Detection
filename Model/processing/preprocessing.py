import tensorflow as tf
import numpy as np

def data_generator(data, sq_len, n_features):
    inputs = tf.zeros((len(data), 0, n_features))

    for i in range(sq_len):
        inputs = tf.concat((inputs, np.array(data.shift(-i))[:,np.newaxis,:-1]), axis = 1)

    return inputs[:-sq_len+1,:,:], tf.convert_to_tensor(data['Class'])[sq_len-1:]

class Transactional_Extension(tf.keras.layers.Layer):
    def __init__(self, per_step: int):
        super(Transactional_Extension, self).__init__()
        self.per_step = per_step

    def call(self, inputs, labels):
        pad = tf.tile(inputs[:, 0:1], [1, self.per_step - 1, 1])
        label_pad = tf.tile(labels[..., tf.newaxis], [1, self.per_step])
        fix_data = tf.concat((pad, inputs), axis = 1)
        blocks = tf.concat([fix_data[:, i:i+self.per_step][:, tf.newaxis, ...] for i in range(self.per_step)], axis = 1)
        return tf.reshape(blocks, shape = [blocks.shape[0] * blocks.shape[1], ] + blocks.shape[2:]), tf.reshape(label_pad, shape = (-1))