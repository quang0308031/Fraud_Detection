import tensorflow as tf

class LessThanConstraint(tf.keras.constraints.Constraint):
    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, clip_value_min=-float('inf'), clip_value_max=self.max_value)

class TH_Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, d):
        super(TH_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.d = d

    def build(self, input_shape):
        self.W_aq = self.add_weight(name='Waq', shape=(self.hidden_dim * 2, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ah = self.add_weight(name='Wah', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.v = self.add_weight(name='v', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        #self._ati_dense = self.add_weight(name='ati', shape=(input_shape[-2], self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, ht, ct, hi):
        q = tf.concat((ht, ct), axis = -1)
        aq = tf.matmul(q, self.W_aq)
        ah = tf.matmul(hi, self.W_ah)

        oti = tf.math.tanh(aq[:, tf.newaxis, ...] + ah + self.ba)
        dims=tf.range(len(tf.shape(oti)))
        perm = tf.concat((dims[:-2], dims[-1:], dims[-2:-1]), axis = 0)
        oti = tf.transpose(oti, perm=perm)

        a = tf.matmul(self.v, oti)[:,0]
        ati = tf.exp(a)
        ati = ati / tf.reduce_sum(ati, axis=1)[:, tf.newaxis, ...]

        #ati = tf.matmul(ati, self._ati_dense)
        e = tf.reduce_sum(ati[..., tf.newaxis] * hi, axis=1)
        return e
    
class TH_InteractionModule(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        super(TH_InteractionModule, self).__init__()

    def build(self, input_shape):
        self.wh = self.add_weight(name='wh', shape = (self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.we = self.add_weight(name='we', shape = (self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.wg = self.add_weight(name='wg', shape = (input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bh = self.add_weight(name='bh', shape = (1, self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, g, ht, e):
        _h = tf.matmul(ht, self.wh)
        _e = tf.matmul(e, self.we)
        _g = tf.matmul(g, self.wg)
        new_h = tf.math.tanh(_h + _e + _g + self.bh)
        return new_h
    
class HAINT_Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, d):
        super(HAINT_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.d = d

    def build(self, input_shape):
        self.W_as = self.add_weight(name='Was', shape=(self.hidden_dim * 2, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_ah = self.add_weight(name='Wah', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.ba = self.add_weight(name='ba', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.v = self.add_weight(name='v', shape=(1, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.W_a = self.add_weight(name='Wa', shape=(self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        #self._ati_dense = self.add_weight(name='ati', shape=(input_shape[-2], self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, ht, ct, hi):
        q = tf.concat((ht, ct), axis = -1)
        a_s = tf.matmul(q, self.W_as)
        ah = tf.matmul(hi, self.W_ah)

        etk = tf.math.tanh(a_s[:, tf.newaxis, ...] + ah + self.ba)
        etk = tf.matmul(etk, self.W_a)

        atk = tf.math.softmax(etk, axis = 1)

        #ati = tf.matmul(ati, self._ati_dense)
        e = tf.reduce_sum(atk * hi, axis=1)
        return 
    
class HAINT_InteractionModule(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        super(HAINT_InteractionModule, self).__init__()

    def build(self, input_shape):
        self.wh = self.add_weight(name='wh', shape = (self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.wp = self.add_weight(name='wp', shape = (input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        self.wg = self.add_weight(name='wg', shape = (self.hidden_dim, self.hidden_dim), initializer="random_uniform", trainable=True)
        self.bh = self.add_weight(name='bh', shape = (1, self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, e, g, ht):
        _h = tf.matmul(ht, self.wh)
        _p = tf.matmul(e, self.wp)
        _g = tf.matmul(g, self.wg)
        new_h = tf.math.tanh(_h + _p + _g + self.bh)
        return new_h