import tensorflow as tf
from .layers import TH_Attention
from .layers import TH_InteractionModule
from .layers import LessThanConstraint
from .layers import HAINT_Attention
from .layers import HAINT_InteractionModule

class TH_cell(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, d, **kwargs):
        super().__init__( **kwargs)
        self.hidden_dim = hidden_dim
        self.attn = TH_Attention(hidden_dim, d)
        self.IM = TH_InteractionModule(hidden_dim)
        self.d = d

    def build(self, input_shape):
        self.W_sh = self.add_weight(name='Wsh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_sx = self.add_weight(name='Wsx', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_st = self.add_weight(name='Wst', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bs = self.add_weight(name='bs', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_fh = self.add_weight(name='Wfh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_fx = self.add_weight(name='Wfx', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_fs = self.add_weight(name='Wfs', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True, constraint=LessThanConstraint(max_value=0.0))
        self.bf = self.add_weight(name='bf', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_ih = self.add_weight(name='Wih', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ix = self.add_weight(name='Wix', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_is = self.add_weight(name='Wis', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bi = self.add_weight(name='bi', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_Th = self.add_weight(name='WTh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_Tx = self.add_weight(name='WTx', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_Ts = self.add_weight(name='WTs', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bT = self.add_weight(name='bT', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_Eh = self.add_weight(name='WEh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_Ex = self.add_weight(name='WEx', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_Es = self.add_weight(name='WEs', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bE = self.add_weight(name='bE', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_oh = self.add_weight(name='Woh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ox = self.add_weight(name='Wox', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_os = self.add_weight(name='Wos', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bo = self.add_weight(name='bo', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, input, h_prev, c_prev, delta_t, mem):
        sh = tf.matmul(h_prev, self.W_sh)
        sx = tf.matmul(input, self.W_sx)
        st = tf.matmul(delta_t, self.W_st)

        s = tf.tanh(sh + sx + st + self.bs)

        fh = tf.matmul(h_prev, self.W_fh)
        fx = tf.matmul(input, self.W_fx)
        fs = tf.matmul(s, self.W_fs)

        f = tf.sigmoid(fh + fx  + fs + self.bf)

        ih = tf.matmul(h_prev, self.W_ih)
        ix = tf.matmul(input, self.W_ix)
        i_s = tf.matmul(s, self.W_is)

        i = tf.sigmoid(ih + ix  + i_s + self.bi)

        Th = tf.matmul(h_prev, self.W_Th)
        Tx = tf.matmul(input, self.W_Tx)
        Ts = tf.matmul(s, self.W_Ts)

        T = tf.sigmoid(Th + Tx  + Ts + self.bT)

        Eh = tf.matmul(h_prev, self.W_Eh)
        Ex = tf.matmul(input, self.W_Ex)
        Es = tf.matmul(s, self.W_Es)

        E = tf.tanh(Eh + Ex  + Es + self.bE)

        new_c = f * c_prev + i * E + T * s

        oh = tf.matmul(h_prev, self.W_oh)
        ox = tf.matmul(input, self.W_ox)
        os = tf.matmul(s, self.W_os)

        o = tf.sigmoid(oh + ox  + os + self.bo)

        _h = o * tf.math.tanh(new_c)

        e = self.attn(_h, new_c, mem)
        new_h = self.IM(input, _h, e)
        return new_h, new_c
    
class UGRNNCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(UGRNNCell, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.node_update_layer = self.add_weight(name='Wox', shape=(self.units + 1, self.units), initializer="random_uniform", trainable=True)
        self.node_update_layer_bias = self.add_weight(name='Woxb', shape=(1, self.units), initializer="random_uniform", trainable=True)
        self.edge_update_layer = self.add_weight(name='Wos', shape=(self.units * 2 + input_shape[1][-1], self.units), initializer="random_uniform", trainable=True)
        self.edge_update_layer_bias = self.add_weight(name='Wosb', shape=(1, self.units), initializer="random_uniform", trainable=True)
        self.aggregation_layer = self.add_weight(name='bo', shape=(self.units + input_shape[1][-1], self.units), initializer="random_uniform", trainable=True)
        self.aggregation_layer_bias = self.add_weight(name='bob', shape=(1, self.units), initializer="random_uniform", trainable=True)
        self.node_update_layer2 = self.add_weight(name='Wox2', shape=(self.units * 2, self.units), initializer="random_uniform", trainable=True)
        self.node_update_layer_bias2 = self.add_weight(name='Woxb2', shape=(1, self.units), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, inputs):
        # Trích xuất các thành phần dữ liệu
        node_states, neighbour_states, edge_features = inputs

        # Cập nhật trạng thái nút
        ns = tf.reduce_mean(neighbour_states, axis=-1)[..., tf.newaxis]
        n_n = tf.concat([node_states, ns], axis=-1)

        new_node_states = tf.matmul(n_n, self.node_update_layer)
        new_node_states = new_node_states + self.node_update_layer_bias

        n_n = tf.concat([node_states, neighbour_states], axis=-1)
        enn = tf.concat([edge_features, n_n], axis=-1)
        # Cập nhật trạng thái cạnh
        new_edge_features = tf.matmul(enn, self.edge_update_layer)
        new_edge_features = new_edge_features + self.edge_update_layer_bias

        ne = tf.concat([neighbour_states, edge_features], axis=-1)
        # Tổng hợp thông tin hàng xóm
        aggregated_neighbours = tf.matmul(ne, self.aggregation_layer)
        aggregated_neighbours = aggregated_neighbours + self.aggregation_layer_bias

        na = tf.concat([new_node_states, aggregated_neighbours], axis=-1)
        new_node_states = tf.matmul(na, self.node_update_layer2)
        new_node_states = new_node_states + self.node_update_layer_bias2

        return new_node_states, new_edge_features
    
class HAINT_cell(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, d, **kwargs):
        super().__init__( **kwargs)
        self.hidden_dim = hidden_dim
        self.attn = HAINT_Attention(hidden_dim, d)
        self.IM = HAINT_InteractionModule(hidden_dim)
        self.d = d

    def build(self, input_shape):
        self.W_fh = self.add_weight(name='Wfh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_fx = self.add_weight(name='Wfx', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ft = self.add_weight(name='Wft', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bf = self.add_weight(name='bf', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_ih = self.add_weight(name='Wih', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ix = self.add_weight(name='Wix', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bi = self.add_weight(name='bi', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_oh = self.add_weight(name='Woh', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ox = self.add_weight(name='Wox', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bo = self.add_weight(name='bo', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)

        self.W_ch = self.add_weight(name='Wch', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_cx = self.add_weight(name='Wcx', shape=(input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bc = self.add_weight(name='bo', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)
    
    def call(self, input, h_prev, c_prev, delta_t, mem):
        fh = tf.matmul(h_prev, self.W_fh)
        fx = tf.matmul(input, self.W_fx)
        ft = tf.matmul(delta_t, self.W_ft)

        f = tf.sigmoid(fh + fx  + ft + self.bf)

        ih = tf.matmul(h_prev, self.W_ih)
        ix = tf.matmul(input, self.W_ix)

        i = tf.sigmoid(ih + ix + self.bi)

        oh = tf.matmul(h_prev, self.W_oh)
        ox = tf.matmul(input, self.W_ox)

        o = tf.sigmoid(oh + ox + self.bo)

        _ch = tf.matmul(h_prev, self.W_ch)
        _cx = tf.matmul(input, self.W_cx)
        _c = tf.tanh(_ch + _cx + self.bc)

        new_c = f * c_prev + i * _c
        _h = o * tf.tanh(new_c)

        g = self.attn(_h, new_c, mem)
        new_h = self.IM(input, g, _h)
        return new_h, new_c



