import numpy as np
import tensorflow as tf
from Transformer_demo import MultiHeadAttention


class SDM(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, short_term_size, item_field_size, *args, **kwargs):
        super(SDM, self).__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.short_term_size = short_term_size
        self.item_field_size = item_field_size
        self.item_dimension = item_field_size * embedding_size
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_size)
        self.flatten_layer_1 = tf.keras.layers.Flatten()
        self.lstm_layer = tf.keras.layers.LSTM(self.item_dimension, return_sequences=True)
        self.mha_layer = MultiHeadAttention(self.item_dimension, item_field_size)
        self.dense_layer_1 = tf.keras.layers.Dense(self.item_dimension)
        self.transform_layers = [tf.keras.layers.Dense(embedding_size) for _ in range(item_field_size)]
        self.flatten_layer_2 = tf.keras.layers.Flatten()
        self.dense_user_layer = tf.keras.layers.Dense(item_field_size * embedding_size)
        self.dense_short_layer = tf.keras.layers.Dense(item_field_size * embedding_size)
        self.dense_long_layer = tf.keras.layers.Dense(item_field_size * embedding_size)


    def call(self, inputs, training=None, mask=None):
        # User Profile
        user_embedding = self.embedding_layer(inputs['user_profile'])
        user_embedding = self.flatten_layer_1(user_embedding)
        # Short Term Interest Embedding
        short_term_interest = self.embedding_layer(inputs['short_term_behavior'])
        short_term_interest = tf.reshape(short_term_interest, (-1, self.short_term_size, self.item_dimension))
        # LSTM
        short_term_interest = self.lstm_layer(short_term_interest)
        # Multi-Head Attention
        short_term_interest = self.mha_layer(short_term_interest)
        # Weighted Short Term Interest
        attention_score = self.dense_layer_1(user_embedding)
        attention_score = tf.matmul(short_term_interest, tf.expand_dims(attention_score, axis=-1))
        attention_score = tf.nn.softmax(attention_score, axis=1)
        short_term_interest = tf.reduce_sum(tf.multiply(short_term_interest, attention_score), axis=1)
        # Long Short Term Interest
        long_term_interest = list()
        for i, group in enumerate(tf.split(inputs['long_term_behavior'], num_or_size_splits=self.item_field_size, axis=-1)):
            # for each set
            subset_interest = list()
            for set_inx, user_vec in zip(tf.split(group, group.shape[0]),
                                         tf.split(user_embedding, user_embedding.shape[0])):
                # for each batch
                uni_set_inx = tf.unique(tf.reshape(set_inx, [-1]))[0]
                set_vec = self.embedding_layer(uni_set_inx)
                user_vec = self.transform_layers[i](user_vec)
                attention_score = tf.nn.softmax(tf.matmul(set_vec, user_vec, transpose_b=True), axis=0)
                attention_interest = tf.matmul(set_vec, attention_score, transpose_a=True)
                subset_interest.append(attention_interest)
            subset_interest = tf.concat(subset_interest, axis=1)
            long_term_interest.append(subset_interest)
        long_term_interest = tf.stack(long_term_interest)
        long_term_interest = tf.transpose(long_term_interest, perm=[2, 0, 1])
        long_term_interest = self.flatten_layer_2(long_term_interest)
        # Objective Interest
        interest_gate = tf.add_n([self.dense_user_layer(user_embedding),
                                  self.dense_short_layer(short_term_interest),
                                  self.dense_long_layer(long_term_interest)])
        interest_gate = tf.nn.sigmoid(interest_gate)
        objective_interest = (1 - interest_gate) * long_term_interest + interest_gate * short_term_interest
        return objective_interest


if __name__ == '__main__':
    batch_size = 16
    user_field_size = 10
    item_field_size = 6
    embedding_size = 8
    feature_size = 5000

    short_term_size = 10  # interactive items for short term interest
    long_term_size = 30  # interactive items for long term interest

    data = dict()
    data['user_profile'] = np.random.randint(0, feature_size, (batch_size, user_field_size))
    data['short_term_behavior'] = np.random.randint(0, feature_size, (batch_size, short_term_size, item_field_size))
    data['long_term_behavior'] = np.random.randint(0, feature_size, (batch_size, long_term_size, item_field_size))
    model = SDM(feature_size, embedding_size, short_term_size, item_field_size)
    objective_interest = model(data)