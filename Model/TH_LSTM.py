import tensorflow as tf
from .processing.preprocessing import Transactional_Extension
from .component.cells import cell

class TH_LSTM(tf.keras.layers.Layer):
    def __init__(self, len: int, per_step: int, k: int):
        super(TH_LSTM, self).__init__()
        self.units = len - k + 1
        self.k = k
        self.per_step = per_step
        self.TE = Transactional_Extension(per_step = per_step, k = k)
        self.norm = tf.keras.layers.LayerNormalization()

    def build(self, input_shape):
        batch = input_shape[0]
        input_shape = [self.TE.k, self.per_step, input_shape[-1]]
        self.w_sh = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Time_aware_input_hidden_state')
        self.w_sx = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Time_aware_input')
        self.w_st = tf.keras.layers.EinsumDense('...xf,xd->...xd', output_shape=input_shape[:-1], bias_axes='xd', name='Time_aware_time_step')
        self.b_s = self.add_weight('Time_aware_bias', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.w_fh = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Forget_gate_hidden_state')
        self.w_fx = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Forget_gate_input')
        self.w_fs = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Forget_gate_time_stamp')
        self.b_f = self.add_weight('Forget_gate_bias', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.w_ih = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Input_gate_hidden_state')
        self.w_ix = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Input_gate_input')
        self.w_is = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Input_gate_time_stamp')
        self.b_i = self.add_weight('Input_gate_bias', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.w_Th = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Time-aware_gate_hidden_state')
        self.w_Tx = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Time-aware_gate_input')
        self.w_Ts = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Time-aware_gate_time_stamp')
        self.b_T = self.add_weight('Time-aware_gate_bias', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.w_elph = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Generated_candidate_cell_state_hidden_state')
        self.w_elpx = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Generated_candidate_cell_state_input')
        self.w_elps = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Generated_candidate_cell_state_time_stamp')
        self.b_elp = self.add_weight('Generated_candidate_cell_state_bias', shape = input_shape, initializer='random_normal', trainable=True)

        self.w_oh = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Output_gate_hidden_state')
        self.w_ox = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Output_gate_input')
        self.w_os = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='Output_gate_time_stamp')
        self.b_o = self.add_weight('Output_gate_bias', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.w_aq = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='aq')
        self.w_ah = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='ah')
        self.b_a = self.add_weight('ba', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.w_h = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='wh')
        self.w_e = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='we')
        self.w_g = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='wg')
        self.b_h = self.add_weight('ba', shape = input_shape[1:], initializer='random_normal', trainable=True)

        self.v = tf.keras.layers.EinsumDense('...gxf,gxd->...gxd', output_shape=input_shape, bias_axes='xd', name='v')

        self.cells = [cell(self.w_sh, self.w_sx, self.w_st, self.b_s,
                           self.w_fh, self.w_fx, self.w_fs, self.b_f,
                           self.w_ih, self.w_ix, self.w_is, self.b_i,
                           self.w_Th, self.w_Tx, self.w_Ts, self.b_T,
                           self.w_elph, self.w_elpx, self.w_elps, self.b_elp,
                           self.w_oh, self.w_ox, self.w_os, self.b_o,
                           self.per_step, self.w_aq, self.w_ah, self.b_a,
                           self.k, self.v, self.w_h, self.w_e, self.w_g, self.b_h)
                      for i in range(self.units)]

        self.h_init = self.add_weight('h_init', shape=[1] + input_shape, initializer='zeros', trainable=False)
        self.c_init = self.add_weight('h_init', shape=[1] + input_shape, initializer='zeros', trainable=False)
        return super().build(input_shape)

    def call(self, data_sq):
        shape = tf.shape(data_sq)
        str_mem = tf.zeros((shape[0], 1, self.TE.k, self.per_step, data_sq.shape[-1]))
        data_split = self.TE(data_sq)
        h = self.h_init
        c = self.c_init

        for i in range(self.units):
            c, h = self.cells[i](data_split[:, i,..., 1:], h, c, data_split[:, i,... , 0], str_mem)
            str_mem = tf.concat((str_mem, h[:, tf.newaxis,...]), axis = 1)
        h = self.norm(h)
        c = self.norm(c)
        return h