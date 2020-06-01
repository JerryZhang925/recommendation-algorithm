import tensorflow as tf
import numpy as np
from Transformer_demo import Encoder, positional_encoding


class Attention(tf.keras.layers.Layer):
    def __init__(self, dimmension, **kwargs):
        """
        :param dimmension:
        :param kwargs:
        """
        super(Attention, self).__init__(**kwargs)
        self.dense_layer = tf.keras.layers.Dense(dimmension, use_bias=False)

    def call(self, query, key, **kwargs):
        """
        :param query:
        :param key:
        :param kwargs:
        :return:
        """
        attention_score = self.dense_layer(query)
        attention_score = tf.expand_dims(attention_score, 1)
        attention_score = tf.multiply(key, attention_score)
        attention_score = tf.reduce_sum(attention_score, axis=2, keepdims=True)
        attention_score = tf.nn.softmax(attention_score, axis=1)
        output = tf.multiply(key, attention_score)
        output = tf.reduce_sum(output, axis=1)
        return output


class DSIN(tf.keras.Model):
    def __init__(self, session_size, sequence_size, user_feature_size, field_size,
                 embedding_ndim, embedding_size, num_heads, *args, **kwargs):
        """
        :param session_size:
        :param sequence_size:
        :param user_feature_size:
        :param field_size:
        :param embedding_ndim:
        :param embedding_size:
        :param num_heads:
        :param args:
        :param kwargs:
        """
        super(DSIN, self).__init__(*args, **kwargs)
        self.interest_ndim = field_size * embedding_ndim
        self.session_size = session_size
        self.sequence_size = sequence_size
        self.user_feature_size = user_feature_size
        self.field_size = field_size
        self.embedding_ndim = embedding_ndim
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        # Embedding
        self.embedding_layer = tf.keras.layers.Embedding(embedding_size, embedding_ndim)
        # Transformer
        self.encoder_layer = Encoder(self.interest_ndim , num_heads)
        # Positional Encoding
        self.position_encode = positional_encoding(self.sequence_size, self.interest_ndim)
        # Attention
        self.transformer_attention = Attention(self.interest_ndim)
        # Bi-LSTM
        lstm = tf.keras.layers.LSTM(self.interest_ndim, return_sequences=True)
        self.bilstm_layer = tf.keras.layers.Bidirectional(lstm, merge_mode='sum')
        # Attention
        self.bilstm_attention = Attention(self.interest_ndim)
        # Dense Layers
        self.dense_layer_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        # Embedding
        behavior_embeddings = self.embedding_layer(inputs['behavior_ids'])  # (batch, session, sequence, field, dimmension)
        behavior_embeddings = tf.reshape(behavior_embeddings, (-1, self.session_size, self.sequence_size, self.interest_ndim))
        user_embeddings = self.embedding_layer(inputs['user_features'])
        user_embeddings = tf.reshape(user_embeddings, (-1, self.user_feature_size * self.embedding_ndim))
        item_embeddings = self.embedding_layer(inputs['target_ids'])
        item_embeddings = tf.reshape(item_embeddings, (-1, self.interest_ndim))
        # Session Interest Extract
        behavior_embeddings = tf.reshape(behavior_embeddings, (-1, self.sequence_size, self.interest_ndim))
        behavior_embeddings = tf.add(behavior_embeddings, self.position_encode)
        session_interest = self.encoder_layer(behavior_embeddings)  # （batch, sequence, interest
        session_interest = tf.reshape(session_interest, (-1, self.session_size, self.sequence_size, self.interest_ndim))
        session_interest = tf.reduce_mean(session_interest, axis=2)
        # Interest Activate
        transformer_interest = self.transformer_attention(item_embeddings, session_interest)
        # Session Interest Interact
        bilstm_interest = self.bilstm_layer(session_interest)
        # Interest Activate
        bilstm_interest = self.bilstm_attention(item_embeddings, bilstm_interest)
        # Dense Layers
        dense_vector = tf.concat([user_embeddings, item_embeddings, transformer_interest, bilstm_interest], 1)
        dense_vector = self.dense_layer_1(dense_vector)
        dense_vector = self.dense_layer_2(dense_vector)
        output = self.output_layer(dense_vector)
        return output


if __name__ == '__main__':
    # model parameters
    params = dict()
    params['session_size'] = 4
    params['sequence_size'] = 10
    params['user_feature_size'] = 8
    params['field_size'] = 3
    params['embedding_ndim'] = 8
    params['embedding_size'] = 5000
    params['num_heads'] = 3
    # generate test data
    batch_size = 16
    data = dict()
    data['behavior_ids'] = np.random.randint(0, 5000, (batch_size, params['session_size'], params['sequence_size'], params['field_size']), np.int32)
    data['user_features'] = np.random.randint(0, 5000, (batch_size, params['user_feature_size']), np.int32)
    data['target_ids'] = np.random.randint(0, 5000, (batch_size, params['field_size']), np.int32)
    # build model
    model = DSIN(**params)
    result = model(data)
