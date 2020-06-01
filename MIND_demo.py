import tensorflow as tf
import numpy as np


class MultiInterestExtractor(tf.keras.layers.Layer):
    def __init__(self, interest_size, interest_ndim, iter_routing=50, **kwargs):
        self.interest_size = interest_size
        self.interest_ndim = interest_ndim
        self.iter_routing = iter_routing
        super(MultiInterestExtractor, self).__init__(**kwargs)

    def build(self, input_shape):
        self.behavior_size = input_shape[1]
        self.behavior_ndim = input_shape[2]
        self.affine_layer = tf.keras.layers.Dense(self.interest_ndim)
        super(MultiInterestExtractor, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch_size = inputs.get_shape()[0]
        # b_ijs.shape == (batch_size, i, j)
        b_ijs = tf.constant(np.random.randn(batch_size, self.behavior_size, self.interest_size), dtype=tf.float32)
        u_js = [] # multi-interest
        for r_iter in range(self.iter_routing):
            w_ijs = tf.nn.softmax(b_ijs, axis=1)  # (batch, i, j)
            w_ij_groups = tf.split(w_ijs, self.interest_size, axis=-1)  # j * (batch, i)
            b_ij_groups = tf.split(b_ijs, self.interest_size, axis=-1)  # j * (batch, i)
            for j in range(self.interest_size):  # for interest j
                c_ij = tf.tile(w_ij_groups[j], [1, 1, self.interest_ndim])
                v_ij = self.affine_layer(inputs)
                z_j = tf.reduce_sum(tf.multiply(c_ij, v_ij), 1)  # weighted sum for behavior
                u_j = self.squash(z_j)  # non-linear "squash"
                b_ij_groups[j] = b_ij_groups[j] + tf.matmul(v_ij, tf.expand_dims(u_j, -1))  # update b
                if r_iter == self.iter_routing - 1:
                    u_js.append(tf.expand_dims(u_j, axis=1))
            b_ijs = tf.concat(b_ij_groups, axis=-1)
        u_js = tf.concat(u_js, axis=1)
        return u_js

    def squash(self, z_j):
        z_j_norm_square = tf.reduce_mean(tf.square(z_j), axis=1, keepdims=True)
        u_j = z_j_norm_square * z_j / ((1 + z_j_norm_square) * tf.sqrt(z_j_norm_square + 1e-9))
        return u_j

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.interest_size, self.interest_ndim)

    def get_config(self):
        config = super(MultiInterestExtractor, self).get_config().copy()
        config['interest_size'] = self.interest_size
        config['interest_ndim'] = self.interest_ndim
        config['behavior_size'] = self.behavior_size
        config['behavior_ndim'] = self.behavior_ndim
        config['iter_routing'] = self.iter_routing
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == '__main__':
    item_embeddings = tf.Variable(np.random.randn(4, 5, 8), dtype=tf.float32)
    target_embedding = tf.Variable(np.random.randn(4, 1, 8), dtype=tf.float32)
    target_embedding = tf.transpose(target_embedding, [0, 2, 1])
    user_profiles = tf.Variable(np.random.randn(4, 7), dtype=tf.float32)
    user_interest = MultiInterestExtractor(3, 6)(item_embeddings)
    user_profiles = tf.tile(tf.expand_dims(user_profiles, axis=1), [1, 3, 1])
    concat_vector = tf.concat([user_profiles, user_interest], -1)
    hidden_vector = tf.keras.layers.Dense(32, activation=tf.nn.relu)(concat_vector)
    multi_interest = tf.keras.layers.Dense(8, activation=tf.nn.relu, name='multi_interest')(hidden_vector)
    # label-aware attention
    attention_score = tf.nn.softmax(tf.math.pow(tf.matmul(multi_interest, target_embedding), 2), axis=1)
    user_embeddings = tf.reduce_sum(tf.multiply(multi_interest, attention_score), axis=1)
