import tensorflow as tf
from .component.cells import TH_cell
from .component.cells import UGRNNCell
from .component.cells import T_cell
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .component.cells import HAINT_cell

def forward(dropout, num_classes):
    return tf.keras.Sequential(
    [
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]
)

@tf.keras.saving.register_keras_serializable()
class TH_LSTM(tf.keras.Model):
    def __init__(self, hidden_dim, num_classes, d, dropout = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.d = d
        self.forward = forward(dropout=dropout, num_classes=num_classes)

    def build(self, input_shape):
        self.hc_init = self.add_weight(name='h_init', shape = (1, self.hidden_dim), initializer='zeros')
        self.cell = TH_cell(self.hidden_dim, self.d)
        self._init = self.add_weight(name='init', shape = (input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, inputs):
        str_mem = tf.ones_like(inputs)
        str_mem = tf.matmul(str_mem, self._init)[:,:-1]

        h = c = self.hc_init
        for i in range(inputs.shape[1]):
            h, c = self.cell(inputs[:,i,..., 1:], h, c, inputs[:, i,... ,0:1], str_mem)
            str_mem = tf.concat((str_mem, h[:, tf.newaxis, ...]), axis = 1)[:,-self.d:,...]

        output = self.forward(h)
        return output
    
@tf.keras.saving.register_keras_serializable()
class A_CNN(tf.keras.Model):
    def __init__(self,num_classes,filters, merge_units, head, dropout = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = head
        self.CNN_blocks = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1DTranspose(filters=filters, kernel_size=3, padding='same'),
                    tf.keras.layers.MaxPool1D(),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding='same'),
                    tf.keras.layers.MaxPool1D(),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(merge_units, activation='relu')
                ]
            )
        for i in range(head)]
        self.merge_layers = tf.keras.layers.Dense(merge_units, activation='relu')
        self.forward = forward(dropout=dropout, num_classes=num_classes)

    def call(self, inputs):
        _dense_output = []
        for i in range(self.head):
            otp = self.CNN_blocks[i](inputs)
            _dense_output.append(otp)
        _dense_output = tf.concat(_dense_output, axis=-1)
        output = self.merge_layers(_dense_output)
        output = self.forward(output)
        return output
    
class stan_2d_model(nn.Module):
    """
    1.attribute embeddig(dimension reduction for one-hot features)
    2.attention
    3.cnn
    4.linear layer
    """

    def __init__(
        self,
        time_windows_dim: int,
        feat_dim: int,
        num_classes: int,
        attention_hidden_dim: int,
        cate_unique_num: list = [1664, 216, 2500],
        filter_sizes: tuple = (2, 2),
        num_filters: int = 64,
        in_channels: int = 1
    ) -> None:
        super().__init__()
        self.time_windows_dim = time_windows_dim
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.attention_hidden_dim = attention_hidden_dim

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # cate embedding layer
        # ['Location','Type','Target']
        # self.cate_emdeds = nn.ModuleList([nn.Embedding(
        #     cate_unique_num[idx] + 1, cate_embed_dim) for idx in range(cate_feat_num)])

        # attention layer
        self.attention_W = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_U = nn.Parameter(torch.Tensor(
            self.feat_dim, self.attention_hidden_dim).uniform_(0., 1.))
        self.attention_V = nn.Parameter(torch.Tensor(
            self.attention_hidden_dim, 1).uniform_(0., 1.))

        # cnn layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=filter_sizes
        )

        # FC layer
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Dropout(0.5),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.ReLU(),
            nn.LazyLinear(self.num_classes)
        )

    def attention_layer(
        self,
        X: torch.Tensor
    ):
        self.output_att = []
        # split along time_windows axis
        input_att = torch.split(X, 1, dim=1)
        for index, x_i in enumerate(input_att):
            x_i = x_i.reshape(-1, self.feat_dim)
            c_i = self.attention(x_i, input_att, index)

            inp = torch.concat([x_i, c_i], axis=1)
            self.output_att.append(inp)

        input_conv = torch.reshape(torch.concat(self.output_att, axis=1),
                                   [-1, self.time_windows_dim, self.feat_dim*2])

        self.input_conv_expanded = torch.unsqueeze(input_conv, 1)

        return self.input_conv_expanded

    def cnn_layer(
        self,
        input: torch.Tensor
    ):
        if len(input.shape) == 3:
            self.input_conv_expanded = torch.unsqueeze(input, 1)
        elif len(input.shape) == 4:
            self.input_conv_expanded = input
        else:
            print("Wrong conv input shape!")

        self.input_conv_expanded = F.relu(self.conv(input))

        return self.input_conv_expanded

    def attention(self, x_i, x, index):
        e_i = []
        c_i = []

        for i in range(len(x)):
            output = x[i]
            output = output.reshape(-1, self.feat_dim)
            att_hidden = torch.tanh(torch.add(torch.matmul(
                x_i, self.attention_W), torch.matmul(output, self.attention_U)))
            e_i_j = torch.matmul(att_hidden, self.attention_V)
            e_i.append(e_i_j)

        e_i = torch.concat(e_i, axis=1)
        # print(f"e_i shape: {e_i.shape}")
        alpha_i = F.softmax(e_i, dim=1)
        alpha_i = torch.split(alpha_i, 1, 1)  # !!!

        for j, (alpha_i_j, output) in enumerate(zip(alpha_i, x)):
            if j == index:
                continue
            else:
                output = output.reshape(-1, self.feat_dim)
                c_i_j = torch.multiply(alpha_i_j, output)
                c_i.append(c_i_j)

        c_i = torch.reshape(torch.concat(c_i, axis=1),
                            [-1, self.time_windows_dim-1, self.feat_dim])
        c_i = torch.sum(c_i, dim=1)
        return c_i

    def forward(self, X_nume):
        # X shape be like: (batch_size, time_windows_dim, feat_dim)
        out = self.attention_layer(X_nume)

        out = self.cnn_layer(out)
        out = self.flatten(out)

        out = self.linears(out)

        return out
    
@tf.keras.saving.register_keras_serializable()
class UGRNNModel(tf.keras.Model):
    def __init__(self, units, num_steps, num_classes, dropout = 0.5, **kwargs):
        super(UGRNNModel, self).__init__(**kwargs)
        self.units = units
        self.num_steps = num_steps
        self.fl = tf.keras.layers.Flatten()
        self.forward = forward(dropout=dropout, num_classes=num_classes)

    def build(self, input_shape):
        self.ugrnn_cell = UGRNNCell(self.units)
        return super().build(input_shape)

    def call(self, inputs):
        # Trích xuất trạng thái ban đầu và các thành phần khác
        initial_node_states = inputs[..., 0]
        neighbour_states = inputs[..., 1:-1]
        edge_features = inputs[..., -1]
        # Lặp qua các bước truyền bá thông tin
        node_states = initial_node_states
        edge_features = edge_features
        node_states = tf.tile(node_states[..., tf.newaxis], [1, 1, self.units])
        edge_features = tf.tile(edge_features[..., tf.newaxis], [1, 1, self.units])
        for i in range(self.num_steps):
            node_states, edge_features = self.ugrnn_cell((node_states, neighbour_states, edge_features))

        output = self.fl(node_states)
        output = self.forward(output)
        return output

@tf.keras.saving.register_keras_serializable()
class HAINT_LSTM(tf.keras.Model):
    def __init__(self, hidden_dim, num_classes, d, dropout=0.5,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.d = d
        self.forward = forward(dropout=dropout, num_classes=num_classes)

    def build(self, input_shape):
        self.hc_init = self.add_weight(name='h_init', shape = (1, self.hidden_dim), initializer='zeros')
        self.cell = HAINT_cell(self.hidden_dim, self.d)
        self._init = self.add_weight(name='init', shape = (input_shape[-1], self.hidden_dim), initializer="random_uniform", trainable=True)
        return super().build(input_shape)

    def call(self, inputs):
        str_mem = tf.ones_like(inputs)
        str_mem = tf.matmul(str_mem, self._init)[:,:-1]

        h = c = self.hc_init
        for i in range(inputs.shape[1]):
            h, c = self.cell(inputs[:,i,..., 1:], h, c, inputs[:, i,... ,0:1], str_mem)
            str_mem = tf.concat((str_mem, h[:, tf.newaxis, ...]), axis = 1)[:,-self.d:,...]
        output = self.forward(h)
        return output

@tf.keras.saving.register_keras_serializable()
class A_RNN(tf.keras.Model):
    def __init__(self,num_classes,units, merge_units, dropout, head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = head
        self.LSTM_blocks = [
            tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(units=units, return_sequences=True),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.LSTM(units=units*2),
                    tf.keras.layers.Dropout(dropout),
                    tf.keras.layers.Dense(merge_units, activation='relu')
                ]
            )
        for i in range(head)]
        self.merge_layers = tf.keras.layers.Dense(merge_units, activation='relu')
        self.forward = forward(dropout=dropout, num_classes=num_classes)

    def call(self, inputs):
        _dense_output = []
        for i in range(self.head):
            otp = self.LSTM_blocks[i](inputs)
            _dense_output.append(otp)
        _dense_output = tf.concat(_dense_output, axis=-1)
        output = self.merge_layers(_dense_output)
        output = self.forward(output)
        return output

@tf.keras.saving.register_keras_serializable()
class T_LSTM(tf.keras.Model):
    def __init__(self, hidden_dim, num_classes, time_interval = 1, dropout=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.it = time_interval
        self.forward = forward(dropout=dropout, num_classes=num_classes)

    def build(self, input_shape):
        self.hc_init = self.add_weight(name='h_init', shape = (1, self.hidden_dim), initializer='zeros')
        self.cell = T_cell(self.hidden_dim, self.it)
        return super().build(input_shape)

    def call(self, inputs):
        h = c = self.hc_init
        for i in range(inputs.shape[1]):
            h, c = self.cell(inputs[:,i,..., 1:], h, c, inputs[:, i,... ,0:1])
        output = self.forward(h)
        return output