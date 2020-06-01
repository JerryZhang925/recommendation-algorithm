import tensorflow as tf
import numpy as np


class BiInteraction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BiInteraction, self).__init__(**kwargs)

    def call(self, x):
        sum_square = tf.square(tf.reduce_sum(x, axis=1))
        square_sum = tf.reduce_sum(tf.square(x), axis=1)
        bi_inter = tf.multiply(0.5, tf.subtract(sum_square, square_sum))
        return bi_inter

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class NFM(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, *args, **kwargs):
        super(NFM, self).__init__(*args, **kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_size)
        self.linear_layer = tf.keras.layers.Embedding(feature_size, 1)
        self.interact_layer = BiInteraction()
        self.dense_layer_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense_layer_2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense_layer_3 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        # Shallow Part
        first_order_cate = self.linear_layer(inputs['category_index'])
        first_order_nume = self.linear_layer(inputs['numerical_index'])
        first_order_nume = tf.multiply(first_order_nume, tf.expand_dims(inputs['numerical_value'], -1))
        first_order_concat = tf.concat([first_order_cate, first_order_nume], axis=1)
        first_order = tf.reduce_sum(first_order_concat, axis=1)
        # BiInteraction Part: Pair-Wise Interaction Layer
        embedding_cate = self.embedding_layer(inputs['category_index'])
        embedding_nume = self.embedding_layer(inputs['numerical_index'])
        embedding_nume = tf.multiply(embedding_nume, tf.expand_dims(inputs['numerical_value'], -1))
        embeddings = tf.concat([embedding_cate, embedding_nume], axis=1)
        second_order = self.interact_layer(embeddings)
        # Deep Part
        dense_vector = self.dense_layer_1(second_order)
        dense_vector = self.dense_layer_2(dense_vector)
        dense_output = self.dense_layer_3(dense_vector)
        # Output
        output = tf.add_n([first_order, dense_output])
        return output


if __name__ == '__main__':
    # model parameters
    params = dict()
    params['feature_size'] = 5000
    params['embedding_size'] = 8
    # generate test data
    batch_size = 16
    category_size = 12
    numerical_size = 10
    data = dict()
    data['category_index'] = np.random.randint(0, 5000, (batch_size, category_size), np.int32)
    data['numerical_index'] = np.random.randint(0, 5000, (batch_size, numerical_size), np.int32)
    data['numerical_value'] = np.random.randn(batch_size, numerical_size).astype(np.float32)
    # build model
    model = NFM(**params)
    result = model(data)
