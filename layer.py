#coding:utf-8
import tensorflow as tf
from collections import defaultdict
#tf.enable_eager_execution()

class Layer(object):
    _name_dict = defaultdict(int)

    def __init__(self, name=None):
        if name is None:
            name = "layer"

        self.name = name + "_" + str(self._name_dict[name] + 1)
        self._name_dict[name] += 1


class MultiHeadAttention(Layer):
    def __init__(self, heads, units, attention_on_itself=True, name='encoder_block'):
        super(MultiHeadAttention, self).__init__(name)
        self.heads = heads
        self.units = units
        self.attention_on_itself = attention_on_itself  # only workable when query==key
        self.dense_layers = [tf.keras.layers.Dense(units) for _ in range(3)]

    def __call__(self, query, memory=None, mask=None, scope='attn'):
        if memory is None:
            memory = query
        #with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
        Q = tf.layers.dense(query, self.units, activation=tf.nn.relu)
        K = tf.layers.dense(memory, self.units, activation=tf.nn.relu)
        V = tf.layers.dense(memory, self.units, activation=tf.nn.relu)

        # Split the matrix to multiple heads and then concatenate to have a larger
        # batch size: [h*batch, q_size/k_size, d_model/num_heads]
        Q_split = tf.concat(tf.split(Q, self.heads, axis=2), axis=0)
        K_split = tf.concat(tf.split(K, self.heads, axis=2), axis=0)
        V_split = tf.concat(tf.split(V, self.heads, axis=2), axis=0)
        if mask is not None:
            mask =tf.tile(mask, [self.heads, 1, 1])
        out = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask=mask)

        # Merge the multi-head back to the original shape
        out = tf.concat(tf.split(out, self.heads, axis=0), axis=2)  # [bs, q_size, d_model]

        # The final linear layer and dropout.
        # out = tf.layers.dense(out, self.d_model)
        # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)

        return out

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Args:
            Q (tf.tensor): of shape (h * batch, q_size, d_model)
            K (tf.tensor): of shape (h * batch, k_size, d_model)
            V (tf.tensor): of shape (h * batch, k_size, d_model)
            mask (tf.tensor): of shape (h * batch, q_size, k_size)
        """

        d = self.units // self.heads
        assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
        out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

        if mask is not None:
            # masking out (0.0) => setting to -inf.
            out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
        #out = tf.layers.dropout(out, training=self._is_training)
        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

        return out

import numpy as np
def positional_embedding(pos,model_size):
    PE=np.zeros((1,model_size))
    for i in range(model_size):
        if i%2==0:
            PE[:,i] = np.sin(pos/10000**(i/model_size))
        else:
            PE[:,i] = np.sin(pos/10000**((i-1)/model_size))
    return PE

def get_position_embedding(max_len,model_size):
    pes =[]
    for i in range(max_len):
        pes.append(positional_embedding(i,model_size))
    pes = np.concatenate(pes,axis=0)
    pes = tf.constant(pes,dtype=tf.float32)
    return pes



def construct_padding_mask(inp):
    """
    Args: Original input of word ids, shape [batch, seq_len]
    Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
    """
    seq_len = tf.shape(inp)[1]
    mask = tf.cast(tf.not_equal(inp, 0), tf.float32)  # mask '<pad>'
    mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])
    return mask
# look_left_only_mask = tf.reshape(tf.tile(tf.expand_dims(tf.linalg.band_part(tf.ones((4, 4)), -1,0),axis=0),[512,1,1]),[8,64,4,4])
# print(look_left_only_mask)
# target=tf.constant([[1,0,0],
#                     [2,2,0],
#                     ])
# print(target.shape)
# masks = tf.tile(tf.expand_dims(tf.linalg.band_part(tf.ones((tf.shape(target)[1], tf.shape(target)[1])), -1, 0), axis=0),
#                 [tf.shape(target)[0], 1, 1]),
#
# print(tf.squeeze(construct_padding_mask(target)*masks,axis=0))
#
# print(masks)
#
# print(tf.tile(masks, [8, 1, 1,1]))
# #
#
# print(masks)