import tensorflow as tf
from .layers import Attention
from .layers import InteractionModule

class cell(tf.keras.layers.Layer):
    def __init__(self, w_sh, w_sx, w_st, b_s,
                 w_fh, w_fx, w_fs, b_f,
                 w_ih, w_ix, w_is, b_i,
                 w_Th, w_Tx, w_Ts, b_T,
                 w_elph, w_elpx, w_elps, b_elp,
                 w_oh, w_ox, w_os, b_o,
                 per_step, w_aq, w_ah, b_a,
                 k, v, wh, we, wg, b_h):
        super(cell, self).__init__()
        self.w_sh = w_sh
        self.w_sx = w_sx
        self.w_st = w_st
        self.b_s = b_s

        self.w_fh = w_fh
        self.w_fx = w_fx
        self.w_fs = w_fs
        self.b_f = b_f

        self.w_ih = w_ih
        self.w_ix = w_ix
        self.w_is = w_is
        self.b_i = b_i

        self.w_Th = w_Th
        self.w_Tx = w_Tx
        self.w_Ts = w_Ts
        self.b_T = b_T

        self.w_elph = w_elph
        self.w_elpx = w_elpx
        self.w_elps = w_elps
        self.b_elp = b_elp

        self.w_oh = w_oh
        self.w_ox = w_ox
        self.w_os = w_os
        self.b_o = b_o

        self.attn = Attention(k, w_aq, w_ah, b_a, v)
        self.IM = InteractionModule(wh, we, wg, b_h)

    def call(self, x_t, prev_h, prev_c, dt, store):
        s_x = self.w_sx(x_t)
        s_t = self.w_st(dt)
        s_h = self.w_sh(prev_h)

        s = tf.math.tanh(s_h + s_x + s_t[..., tf.newaxis] + self.b_s)

        f_x = self.w_fx(x_t)
        f_t = self.w_fs(s)
        f_h = self.w_fh(prev_h)

        f = tf.math.sigmoid(f_h + f_x + f_t + self.b_f)

        i_x = self.w_ix(x_t)
        i_t = self.w_is(s)
        i_h = self.w_ih(prev_h)

        i = tf.math.sigmoid(i_h + i_x + i_t + self.b_i)

        T_x = self.w_Tx(x_t)
        T_t = self.w_Ts(s)
        T_h = self.w_Th(prev_h)

        T = tf.math.sigmoid(T_h + T_x + T_t + self.b_T)

        elp_x = self.w_elpx(x_t)
        elp_t = self.w_elps(s)
        elp_h = self.w_elph(prev_h)

        elp = tf.math.tanh(elp_h + elp_x + elp_t + self.b_elp)

        new_c = f * prev_c + i * elp + T * s

        o_x = self.w_ox(x_t)
        o_t = self.w_os(s)
        o_h = self.w_oh(prev_h)

        o = tf.math.sigmoid(o_h + o_x + o_t + self.b_o)

        h = o * tf.math.sigmoid(new_c)

        attn_score = self.attn(h, new_c, store)
        new_h = self.IM(h, attn_score, x_t)
        return new_c, new_h
