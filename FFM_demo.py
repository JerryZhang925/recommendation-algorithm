import numpy as np
import tensorflow as tf


class FFM(tf.keras.Model):
    def __init__(self, feature_size, field_size, dimension, *args, **kwargs):
        super(FFM, self).__init__(*args, **kwargs)
        weight_shape = [feature_size, field_size, dimension]
        self.embedding_weights = self.add_weight(shape=weight_shape, dtype=tf.float32)
        pair_wise_index = [[i, j] for i in range(field_size - 1) for j in range(i + 1, field_size)]  # cross field index
        self.pair_wise_index = tf.constant(pair_wise_index, dtype=tf.int32)
        right_field_index, left_field_index = tf.split(self.pair_wise_index, 2, axis=1)
        self.left_field_index = tf.expand_dims(left_field_index, axis=0)
        self.right_field_index = tf.expand_dims(right_field_index, axis=0)

    def call(self, inputs, training=None, mask=None):
        pair_feature_index = tf.gather(inputs, self.pair_wise_index, axis=1)
        left_feature, right_feature = tf.split(pair_feature_index, 2, axis=2)
        left_field = tf.tile(self.left_field_index, [left_feature.shape[0], 1, 1])
        right_field = tf.tile(self.right_field_index, [right_feature.shape[0], 1, 1])
        left_index = tf.concat([left_feature, left_field], axis=-1)  # (feature_index, target_field_index)
        right_index = tf.concat([right_feature, right_field], axis=-1)  # (feature_index, target_field_index)
        left_embedding = tf.gather_nd(self.embedding_weights, left_index)
        right_embedding = tf.gather_nd(self.embedding_weights, right_index)
        cross_value = tf.reduce_sum(tf.multiply(left_embedding, right_embedding), axis=-1)
        output = tf.reduce_sum(cross_value, axis=-1, keepdims=True)
        return output


if __name__ == '__main__':
    # model parameters
    params = dict()
    params['feature_size'] = 5000
    params['field_size'] = 12
    params['dimension'] = 8
    # generate test data
    batch_size = 16
    data = np.random.randint(0, 5000, (batch_size, params['field_size']), np.int32)
    # build model
    model = FFM(**params)
    result = model(data)