import tensorflow as tf
import numpy as np


class FwFM(tf.keras.Model):
    def __init__(self, feature_size, field_size, dimension, *args, **kwargs):
        super(FwFM, self).__init__(*args, **kwargs)
        self.feature_size = feature_size
        self.field_size = field_size
        self.dimension = dimension
        cross_size = int(field_size * (field_size - 1) / 2)
        self.embedding_weights = self.add_weight(name='embedding_weights', shape=(feature_size, dimension),
                                                 dtype=tf.float32)
        self.field_weights = self.add_weight(name='field_weights', shape=(cross_size, 1), dtype=tf.float32)
        self.linear_weights = self.add_weight(name='linear_weights', shape=(feature_size,), dtype=tf.float32)
        self.bias_weight = self.add_weight(name='bias_weight', initializer=tf.initializers.zeros(), dtype=tf.float32)
        pair_wise_index = [[i, j] for i in range(field_size - 1) for j in range(i + 1, field_size)]  # cross field index
        self.pair_wise_index = tf.constant(pair_wise_index, dtype=tf.int32)

    def call(self, inputs, training=None, mask=None):
        # First Order
        first_order = tf.gather(self.linear_weights, inputs)
        first_order = tf.reduce_sum(first_order, keepdims=True, axis=1)
        # Bias
        bias = tf.multiply(tf.ones_like(first_order), self.bias_weight)
        # Second Order
        pair_feature_index = tf.gather(inputs, self.pair_wise_index, axis=1)  # (batch_size, cross_items, pair)
        embeddings = tf.gather(self.embedding_weights, pair_feature_index)  # (batch_size, cross_items, pair, dimension)
        second_items = tf.reduce_sum(tf.reduce_prod(embeddings, axis=2), axis=-1)
        second_order = tf.matmul(second_items, self.field_weights)
        # Concat
        output = tf.add_n([bias, first_order, second_order])
        return output


if __name__ == '__main__':
    # model parameters
    params = dict()
    params['feature_size'] = 5000
    params['field_size'] = 16
    params['dimension'] = 8
    # generate test data
    batch_size = 16
    data = np.random.randint(0, 5000, (batch_size, params['field_size']), np.int32)
    # build model
    model = FwFM(**params)
    result = model(data)