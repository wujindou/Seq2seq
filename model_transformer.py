#coding:utf-8
import tensorflow as tf
from layer import get_position_embedding,MultiHeadAttention
import numpy as np
class Transformer(object):

    def build_inputs(self, config):
        self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs')
        self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length')
        self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets')
        self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length')
        self.config = config
        self.encoder_num_layers = 4
        self.decoder_num_layers = 4
        self.learning_rate = 1e-4
        self.head_num = 8
        self.model_size = 512
        self.PAD_ID = 0
        self.max_length = 45
        self.pes = get_position_embedding(self.max_length,self.model_size)


    def build_loss(self, logits):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.seq_targets,
            logits=self.logits,
        )
        self.loss = tf.reduce_mean(loss)


    def build_optim(self, loss, lr):
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    def encoder(self):

        with tf.variable_scope("embedding"):
            encoder_embedding = tf.Variable(tf.random_uniform([self.config.source_vocab_size, self.model_size]),
                                            dtype=tf.float32, name='encoder_embedding')
            encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)
            encoder_pos_emb =self.pes[:tf.shape(self.seq_inputs)[1],:]
            encoder_inputs_embedded+=encoder_pos_emb
        sub_in = encoder_inputs_embedded
        attention = [MultiHeadAttention(self.head_num,self.model_size,name="attention_"+str(i)) for i in range(self.encoder_num_layers)]
        attention_norm = [tf.keras.layers.BatchNormalization() for _ in range(self.encoder_num_layers)]
        dense_1 = [tf.keras.layers.Dense(4*self.model_size) for _ in range(self.encoder_num_layers)]
        dense_2 = [tf.keras.layers.Dense(self.model_size) for _ in range(self.encoder_num_layers)]
        ffn_norm = [tf.keras.layers.BatchNormalization()for _ in range(self.encoder_num_layers)]

        self.input_mask = self.construct_padding_mask(self.seq_inputs)
        for i in range(self.encoder_num_layers):
            sub_out = attention[i](sub_in,sub_in,self.input_mask)
            sub_out+=sub_in
            sub_out = attention_norm[i](sub_out)
            ffn_in = sub_out
            ffn_out = dense_2[i](dense_1[i](ffn_in))
            ffn_out += ffn_in
            ffn_out = ffn_norm[i](ffn_out)
            sub_in = ffn_out

        self.encoder_out = ffn_out






    def decoder(self):
        with tf.variable_scope("decoder-embedding"):
            tokens_go = tf.ones([self.config.batch_size], dtype=tf.int32, name='tokens_GO') * (self.tokenizer.convert_tokens_to_ids(["[unused1]"])[0])

            decoder_inputs = tf.concat([tf.reshape(tokens_go, [-1, 1]), self.seq_targets[:, :-1]], 1)

            decoder_embedding = tf.Variable(
                tf.random_uniform([self.config.target_vocab_size, self.model_size]),
                dtype=tf.float32, name='decoder_embedding')
            decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
            decoder_pos_emb = self.pes[:tf.shape(self.seq_targets)[1], :]
            decoder_inputs_embedded += decoder_pos_emb
        bot_sub_in = decoder_inputs_embedded

        attention_bot = [MultiHeadAttention(self.head_num,self.model_size) for _ in range(self.decoder_num_layers)]
        attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(self.decoder_num_layers)]
        attention_mid = [MultiHeadAttention(self.head_num, self.model_size) for _ in range(self.decoder_num_layers)]
        attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(self.decoder_num_layers)]
        dense_1 = [tf.keras.layers.Dense(4 * self.model_size) for _ in range(self.encoder_num_layers)]
        dense_2 = [tf.keras.layers.Dense(self.model_size) for _ in range(self.encoder_num_layers)]
        ffn_norm = [tf.keras.layers.BatchNormalization()for _ in range(self.encoder_num_layers)]

        # The target mask hides both <pad> and future words.
        target_mask = self.construct_padding_mask(self.seq_targets)
        target_mask *= self.construct_autoregressive_mask(self.seq_targets)
        target_mask = tf.squeeze(target_mask,axis=0)

        input_mask2 = tf.expand_dims(tf.cast(tf.sequence_mask(self.seq_inputs_length, maxlen=tf.reduce_max(self.seq_inputs_length)), dtype=tf.float32),axis=1)#input_target_mask = self.construct_padding_mask2(self.seq_targets_length,self.seq_inputs_length)

        for i in range(self.decoder_num_layers):


            bot_sub_out = attention_bot[i](bot_sub_in,bot_sub_in,target_mask)
            bot_sub_out +=bot_sub_in
            bot_sub_out = attention_bot_norm[i](bot_sub_out)

            mid_sub_in = bot_sub_out
            mid_sub_out = attention_mid[i](mid_sub_in,self.encoder_out,input_mask2)

            mid_sub_out +=mid_sub_in
            mid_sub_out = attention_mid_norm[i](mid_sub_out)

            ffn_in  = mid_sub_out
            ffn_out = dense_2[i](dense_1[i](ffn_in))
            ffn_out+=ffn_in
            ffn_out = ffn_norm[i](ffn_out)
        dense = tf.layers.Dense(self.config.target_vocab_size)
        self.logits = dense(ffn_out)
        self.out =  tf.argmax(self.logits, 2)

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.seq_targets,
        #     logits=self.logits,
        # )
        # self.loss = tf.reduce_mean(loss)
        sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.seq_targets,
                                                     weights=sequence_mask)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


    def __init__(self, config,tokenizer):
        self.tokenizer = tokenizer
        self.build_inputs(config)
        self.encoder()
        self.decoder()

    def construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = tf.shape(inp)[1]
        mask = tf.cast(tf.not_equal(inp, self.PAD_ID), tf.float32)  # mask '<pad>'
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])
        return mask
    def construct_autoregressive_mask(self,target):
        #batch_size, seq_len = target.shape.as_list()

        masks = tf.tile(tf.expand_dims(tf.linalg.band_part(tf.ones((tf.shape(target)[1], tf.shape(target)[1])), -1, 0), axis=0), [tf.shape(target)[0], 1, 1]),

        # tri_matrix = np.zeros((seq_len, seq_len))
        # tri_matrix[np.tril_indices(seq_len)] = 1
        #
        # mask = tf.convert_to_tensor(tri_matrix, dtype=tf.float32)
        # masks = tf.tile(tf.expand_dims(mask, 0), [tf.shape(target)[0], 1, 1])  # copies
        return masks

    def construct_padding_mask2(self,inp,dec):
        mask = tf.expand_dims(tf.sequence_mask(self.seq_inputs_length, tf.shape(inp)[1], dtype=tf.float32), axis=2) * \
               tf.expand_dims(tf.sequence_mask(self.seq_targets_length, tf.shape(dec)[1], dtype=tf.float32), axis=1)
        return mask


# if __name__=='__main__':
#     tf.enable_eager_execution()

