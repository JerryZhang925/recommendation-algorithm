import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, ndim_model, num_head, **kwargs):
        self.ndim_model = ndim_model
        self.num_head = num_head
        assert ndim_model % num_head == 0
        self.depth = ndim_model // num_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight_query = tf.keras.layers.Dense(self.ndim_model, name='weight_query')
        self.weight_key = tf.keras.layers.Dense(self.ndim_model, name='weight_key')
        self.weight_value = tf.keras.layers.Dense(self.ndim_model, name='weight_value')
        self.dense_layer = tf.keras.layers.Dense(self.ndim_model, name='dense_layer')

        super(MultiHeadAttention, self).build(input_shape)

    def split_multi_head(self, inputs):
        new_shape = inputs.get_shape().as_list()[:2] + [self.num_head, self.depth]
        output = tf.reshape(inputs, new_shape)
        output = tf.transpose(output, [0, 2, 1, 3])
        return output

    def scaled_dot_product_attention(self, query, key, value):
        dimension = tf.cast(query.get_shape()[-1], tf.float32)
        query_key = tf.matmul(query, key, transpose_b=True)
        attention_logits = tf.math.divide(query_key, tf.math.sqrt(dimension))
        attention_weights = tf.nn.softmax(attention_logits, -1)
        output = tf.matmul(attention_weights, value)
        return output

    def call(self, inputs, **kwargs):
        num_sequence = inputs.get_shape()[1]
        query = self.weight_query(inputs)
        key = self.weight_key(inputs)
        value = self.weight_value(inputs)
        query = self.split_multi_head(query)
        key = self.split_multi_head(key)
        value = self.split_multi_head(value)
        scaled_attention = self.scaled_dot_product_attention(query, key, value)
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        scaled_attention = tf.reshape(scaled_attention, (-1, num_sequence, self.ndim_model))
        output = self.dense_layer(scaled_attention)
        return output

    def compute_output_shape(self, input_shape):
        output = list(input_shape)
        output[-1] = self.ndim_model
        return output

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config().copy()
        config['ndim_model'] = self.ndim_model
        config['num_head'] = self.num_head
        config['depth'] = self.depth
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], dtype=tf.float32, initializer='ones')
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], dtype=tf.float32, initializer='zeros')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        std = tf.math.sqrt(var)
        output = self.gamma * (inputs - mean) / (std + self.epsilon) + self.beta
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(LayerNormalization, self).get_config().copy()
        config['epsilon'] = self.epsilon
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, ndim_model, num_head, **kwargs):
        self.ndim_model = ndim_model
        self.num_head = num_head
        super(Encoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_layer = MultiHeadAttention(self.ndim_model, self.num_head)
        self.normalize_layer_1 = LayerNormalization()
        self.normalize_layer_2 = LayerNormalization()
        self.dense_layer_1 = tf.keras.layers.Dense(self.ndim_model, activation=tf.nn.relu)
        self.dense_layer_2 = tf.keras.layers.Dense(self.ndim_model)
        super(Encoder, self).build(input_shape)

    def call(self, inputs):
        attention_output = self.attention_layer(inputs)
        residual_output_1 = tf.add(inputs, attention_output)
        normalize_output = self.normalize_layer_1(residual_output_1)
        dense_output = self.dense_layer_1(normalize_output)
        dense_output = self.dense_layer_2(dense_output)
        residual_output_2 = tf.add(normalize_output, dense_output)
        output = self.normalize_layer_2(residual_output_2)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(Encoder, self).get_config().copy()
        config['ndim_model'] = self.ndim_model
        config['num_head'] = self.num_head
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def get_angles(position, i, ndim_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(ndim_model))
    return position * angle_rates


def positional_encoding(position, ndim_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(ndim_model)[np.newaxis,:], ndim_model)
    # 2i -> sin
    sines = np.sin(angle_rads[:, 0::2])
    # 2i+1 -> cos
    cones = np.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cones], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
    return pos_encoding


