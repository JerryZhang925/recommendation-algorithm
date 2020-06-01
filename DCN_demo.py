import tensorflow as tf
import numpy as np


class CrossNetwork(tf.keras.layers.Layer):
    def __init__(self, cross_size=3, **kwargs):
        self.cross_size = cross_size
        super(CrossNetwork, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights_dict = dict()
        for i in range(self.cross_size):
            self.weights_dict['weight_%d'%(i+1)] = self.add_weight(name='weight_%d'%(i+1), shape=(input_shape[-1], 1),
                                                                   initializer='glorot_normal', trainable=True)
            self.weights_dict['bias_%d'%(i+1)] = self.add_weight(name='bias_%d'%(i+1), shape=(input_shape[-1], 1),
                                                                 initializer='zeros', trainable=True)
        super(CrossNetwork, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=2)
        xl = inputs
        for i in range(self.cross_size):
            cross_value = tf.matmul(inputs, xl, transpose_b=True)
            cross_value = tf.matmul(cross_value, self.weights_dict['weight_%d'%(i+1)])
            cross_value = tf.add(cross_value, self.weights_dict['bias_%d'%(i+1)])
            xl = tf.add(cross_value, xl)
        xl = tf.squeeze(xl, axis=2)
        return xl

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({'cross_size': self.cross_size})
        return config


class DCN(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, *args, **kwargs):
        super(DCN, self).__init__(*args, **kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_size)
        self.cross_layer = CrossNetwork()
        self.dense_layer_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense_layer_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense_layer_3 = tf.keras.layers.Dense(64)
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        # Embedding
        embedding_cate = self.embedding_layer(inputs['category_index'])
        embedding_nume = self.embedding_layer(inputs['numerical_index'])
        embedding_nume = tf.multiply(embedding_nume, tf.expand_dims(inputs['numerical_value'], -1))
        embeddings = tf.concat([embedding_cate, embedding_nume], axis=1)
        embeddings = tf.reshape(embeddings, (-1, embeddings.shape[1] * embeddings.shape[2]))
        # Cross Part
        cross_output = self.cross_layer(embeddings)
        # Deep Part
        dense_vector = self.dense_layer_1(embeddings)
        dense_vector = self.dense_layer_2(dense_vector)
        deep_output = self.dense_layer_3(dense_vector)
        # Output
        concat_vector = tf.concat([cross_output, deep_output], axis=1)
        output = self.output_layer(concat_vector)
        return output


if __name__ == '__main__':
    # model parameters
    params = dict()
    params['feature_size'] = 5000
    params['embedding_size'] = 8
    # generate test data
    batch_size = 16
    category_size = 10
    numerical_size = 6
    data = dict()
    data['category_index'] = np.random.randint(0, 5000, (batch_size, category_size), np.int32)
    data['numerical_index'] = np.random.randint(0, 5000, (batch_size, numerical_size), np.int32)
    data['numerical_value'] = np.random.randn(batch_size, numerical_size).astype(np.float32)
    # build model
    model = DCN(**params)
    result = model(data)

