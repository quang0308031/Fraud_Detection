import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, k, w_aq, w_ah, b_a, v):
        self.w_aq = w_aq
        self.w_ah = w_ah
        self.b_a = b_a
        self.k = k
        self.v = v
        super(Attention, self).__init__()

    def call(self, ht, ct, hi):
        q = ht[:,tf.newaxis,...] + ct[:,tf.newaxis,...]
        aq = self.w_aq(q)
        ah = self.w_ah(hi[:, -self.k:])
        oti = tf.math.tanh(aq + ah + self.b_a)
        alpha = self.v(oti)
        alpha_exp = tf.math.exp(alpha)
        alphati = alpha_exp / tf.math.reduce_sum(alpha_exp)
        e = alphati * hi[:, -self.k:]
        e_exp = tf.reduce_sum(tf.math.exp(e), axis=1)
        return e_exp / tf.reduce_sum(e_exp)
    
class InteractionModule(tf.keras.layers.Layer):
    def __init__(self, wh, we, wg, b_h):
        self.wh = wh
        self.we = we
        self.wg = wg
        self.b_h = b_h
        super(InteractionModule, self).__init__()

    def call(self, ht, e, g):
        _h = self.wh(ht)
        _e = self.we(e)
        _g = self.wg(g)
        new_h = tf.math.tanh(_h + _e + _g + self.b_h)
        return new_h