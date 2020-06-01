import tensorflow as tf
import numpy as np


class SelfAttentionPooling(tf.keras.layers.Layer):
    def __init__(self, embedding_size, embedding_ndim, sequence_max, hidden_ndim=64, **kwargs):
        self.embedding_size = embedding_size
        self.embedding_ndim = embedding_ndim
        self.sequence_max = sequence_max
        self.hidden_ndim = hidden_ndim
        super(SelfAttentionPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding_layer = tf.keras.layers.Embedding(self.embedding_size, self.embedding_ndim,
                                                         embeddings_regularizer=tf.keras.regularizers.l2)
        self.dense_layer = tf.keras.layers.Dense(self.hidden_ndim, activation=tf.nn.tanh,
                                                 kernel_regularizer=tf.keras.regularizers.l2)
        self.dense_output = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2)
        self.dropout_layer = tf.keras.layers.Dropout(0.5)
        super(SelfAttentionPooling, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        # embedding
        embeddings = self.embedding_layer(inputs[0])
        #  self-attention
        hidden_weights = self.dense_layer(embeddings)
        hidden_weights = self.dropout_layer(hidden_weights, training=training)
        attention_weights = self.dense_output(hidden_weights)
        # mask
        masks = tf.sequence_mask(inputs[1], self.sequence_max)
        masks = tf.expand_dims(masks, -1)
        paddings = tf.cast(tf.ones_like(masks), tf.float32) * float('-inf')
        # attention weight
        attention_weights = tf.where(masks, attention_weights, paddings)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        # weight pooling
        attention_embedding = tf.multiply(embeddings, attention_weights)
        attention_embedding = tf.reduce_sum(attention_embedding, 1)
        return attention_embedding

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embedding_ndim)

    def get_config(self):
        config = super(SelfAttentionPooling, self).get_config()
        config['embedding_size'] = self.embedding_size
        config['embedding_ndim'] = self.embedding_ndim
        config['sequence_max'] = self.sequence_max
        config['hidden_ndim'] = self.hidden_ndim
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class InteractionCube(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InteractionCube, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InteractionCube, self).build(input_shape)

    def call(self, inputs):
        embeddings = tf.transpose(inputs, [1, 0, 2])
        pair_wise_index = [[i, j] for i in range(inputs.shape[1] - 1) for j in range(i + 1, inputs.shape[1])]
        pair_embeddings = tf.nn.embedding_lookup(embeddings, pair_wise_index)  # (cross, 2, batch, dimension)
        pair_embeddings = tf.transpose(pair_embeddings, [2, 0, 1, 3])  # (batch, cross, 2, dimension)
        split_a, split_b = tf.unstack(pair_embeddings, axis=2)  # (batch, cross, dimension)
        split_a = tf.expand_dims(split_a, axis=-1)  # (batch, cross, dimension, 1)
        split_b = tf.expand_dims(split_b, axis=-2)  # (batch, cross, 1, dimension)
        outer_product = tf.matmul(split_a, split_b)  # (batch, cross, dimension, dimension)
        outer_product = tf.transpose(outer_product, [0, 2, 3, 1])
        return outer_product

    def get_config(self):
        config = super(InteractionCube, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_model(field_size, feature_size=5000, embedding_size=64):
    input_index = tf.keras.layers.Input(shape=(10,), dtype=tf.int32)
    embeddings = tf.keras.layers.Embedding(feature_size, embedding_size)(input_index)
    interact_embeddings = InteractionCube()(embeddings)
    feature_map = tf.transpose(interact_embeddings, [0, 3, 1, 2])
    feature_map = tf.expand_dims(feature_map, -1) # set channels is 1
    # Conv3D(channels_last):
    # input-(batch, conv_dim1, conv_dim2, conv_dim3, channels)
    # output-(batch, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)
    feature_map = tf.keras.layers.Conv3D(32, (14, 2, 2), strides=(1, 2, 2), data_format='channels_last',
                                         activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    feature_map = tf.keras.layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last',
                                         padding='same', activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    feature_map = tf.keras.layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last',
                                         padding='same', activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    feature_map = tf.keras.layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last',
                                         padding='same', activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    feature_map = tf.keras.layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last',
                                         padding='same', activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    feature_map = tf.keras.layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), data_format='channels_last',
                                         padding='same', activation=tf.nn.relu,
                                         kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    feature_map = tf.keras.layers.Flatten()(feature_map)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.l2)(feature_map)
    model = tf.keras.Model(inputs=input_index, outputs=output)
    return model

if __name__ == '__main__':
    batch_size = 16
    field_size = 10
    data = np.random.randint(0, 5000, (batch_size, field_size))
    model = build_model(field_size)
    result = model(data)

