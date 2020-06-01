import numpy as np
import tensorflow as tf


class Dice(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, axis=-1, **kwargs):
        self.epsilon = epsilon
        self.axis = axis
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(), dtype=tf.float32, initializer='zeros')
        self.bn = tf.keras.layers.BatchNormalization(axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        super(Dice, self).build(input_shape)

    def call(self, inputs, training, **kwargs):
        norm_data = tf.nn.sigmoid(self.bn(inputs, training=training))
        output = self.alpha * (1.0 - norm_data) * inputs + norm_data * inputs
        return output


class ActivationUnit(tf.keras.layers.Layer):
    def __init__(self, sequence_size, field_size, **kwargs):
        """
        :param sequence_size:
        :param field_size:
        :param kwargs:
        """
        self.sequence_size = sequence_size
        self.field_size = field_size
        self.dense_layers = dict()
        for inx in range(self.field_size):
            self.dense_layers['hidden_layer_%d' % inx] = tf.keras.layers.Dense(36)
            self.dense_layers['activate_layer_%d' % inx] = Dice()
            self.dense_layers['output_layer_%d' % inx] = tf.keras.layers.Dense(1)
        super(ActivationUnit, self).__init__(**kwargs)

    def call(self, query, key, sequence_num, training, **kwargs):
        """
        :param query:
        :param key:
        :param sequence_num:
        :param kwargs:
        :return:
        """
        query = tf.tile(tf.expand_dims(query, 2), [1, 1, self.sequence_size, 1])
        delta = tf.subtract(query, key)
        concat_vector = tf.concat([key, query, delta], axis=-1)
        # FCs
        dense_vector = list()
        for inx, group in enumerate(tf.split(concat_vector, self.field_size, axis=1)):
            # for each fields
            field_vector = self.dense_layers['hidden_layer_%d' % inx](group)
            field_vector = self.dense_layers['activate_layer_%d' % inx](field_vector, training)
            field_vector = self.dense_layers['output_layer_%d' % inx](field_vector)
            dense_vector.append(field_vector)
        dense_vector = tf.concat(dense_vector, 1) # (batch, field, sequence, 1)
        dense_vector = tf.squeeze(dense_vector, axis=-1) # (batch, field, sequence)
        # Attention
        masks = tf.sequence_mask(sequence_num, self.sequence_size)  # (batch, field, sequence)
        paddings = tf.cast(tf.ones_like(masks), tf.float32) * float('-inf')
        query_attention = tf.where(masks, dense_vector, paddings)
        query_attention = tf.nn.softmax(query_attention, axis=-1) # (batch, field, sequence)
        query_attention = tf.expand_dims(query_attention, axis=-1) # (batch, field, sequence, 1)
        output = tf.reduce_sum(tf.multiply(query_attention, key), axis=2)
        return output


class DIN(tf.keras.models.Model):
    def __init__(self, embedding_size, embedding_ndim, sequence_size, field_size, **kwargs):
        """
        :param embedding_size:
        :param embedding_ndim:
        :param sequence_size:
        :param field_size:
        :param kwargs:
        """
        super(DIN, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.embedding_ndim = embedding_ndim
        self.sequence_size = sequence_size
        self.field_size = field_size

        self.embedding_layer = tf.keras.layers.Embedding(embedding_size, embedding_ndim)
        self.activate_layer = ActivationUnit(sequence_size, field_size)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer_1 = tf.keras.layers.Dense(200)
        self.activate_layer_1 = Dice()
        self.dense_layer_2 = tf.keras.layers.Dense(80)
        self.activate_layer_2 = Dice()
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:
        :param training:
        :param mask:
        :return:
        """
        candidate_embeddings = self.embedding_layer(inputs['candidate_ids'])  # (batch, field, dimension)
        behavior_embeddings = self.embedding_layer(inputs['behavior_ids'])  # (batch, field, sequence, dimension)
        user_embeddings = self.embedding_layer(inputs['user_features'])
        context_embeddings = self.embedding_layer(inputs['context_features'])
        # Attention
        interest_vector = self.activate_layer(candidate_embeddings, behavior_embeddings, inputs['sequence_num'], training)
        # Concat
        concat_embedding = tf.concat([user_embeddings, context_embeddings, candidate_embeddings, interest_vector], axis=1)
        concat_embedding = self.flatten_layer(concat_embedding)
        # FC Layer
        dense_vector = self.dense_layer_1(concat_embedding)
        dense_vector = self.activate_layer_1(dense_vector, training)
        dense_vector = self.dense_layer_2(dense_vector)
        dense_vector = self.activate_layer_2(dense_vector, training)
        output = self.output_layer(dense_vector)
        return output


if __name__ == '__main__':
    # model parameters
    batch_size = 16
    field_size = 3
    sequence_size = 10
    user_feature_size = 8
    cont_feature_size = 5
    embedding_ndim = 8
    embedding_size = 5000
    # generate test data
    data = dict()
    data['candidate_ids'] = np.random.randint(0, 5000, (batch_size, field_size), np.int32)
    data['behavior_ids'] = np.random.randint(0, 5000, (batch_size, field_size, sequence_size), np.int32)
    data['user_features'] = np.random.randint(0, 5000, (batch_size, user_feature_size), np.int32)
    data['context_features'] = np.random.randint(0, 5000, (batch_size, cont_feature_size), np.int32)
    data['sequence_num'] = np.random.randint(1, sequence_size, (batch_size, field_size), np.int32)
    # build model
    model = DIN(embedding_size, embedding_ndim, sequence_size, field_size)
    result = model(data)