import tensorflow as tf
import numpy as np


class CIN(tf.keras.layers.Layer):
    """
    Compressed Interaction Network
    """
    def __init__(self, D, Hk, k=3, **kwargs):
        self.D = D
        self.Hk = Hk
        self.k = k
        super(CIN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.Hk * self.k)

    def build(self, input_shape):
        m = input_shape[1] # (batch, m, D)
        self.filte_dict = dict()
        # (m, m, 1, Hk_1)
        self.filte_dict['filter_0'] = self.add_weight(name='filter_0', shape=(m, m, 1, self.Hk),
                                                      dtype=tf.float32, initializer='glorot_normal')
        for i in range(1, self.k):
            # (m, Hk, 1, Hk_1)
            self.filte_dict['filter_%d' % i] = self.add_weight(name='filter_%d' % i, shape=(m, self.Hk, 1, self.Hk),
                                                               dtype=tf.float32, initializer='glorot_normal')
        super(CIN, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x0 = inputs # (batch, m, D)
        xk = inputs
        cin_output = list()
        for i in range(self.k):
            # Step 1
            split_x0 = tf.split(x0, self.D, 2)
            split_xk = tf.split(xk, self.D, 2)
            zk_1 = tf.matmul(split_x0, split_xk, transpose_b=True)  # (D, batch, m, Hk)
            zk_1 = tf.transpose(zk_1, [1, 2, 3, 0])  # (batch, m, Hk, D)
            # Step 2
            xk_1 = list()
            for d_zk1 in tf.split(zk_1, self.D, 3):  # at the dimension of D
                xk_1.append(tf.nn.conv2d(d_zk1, self.filte_dict['filter_%d' % i], 1, 'VALID'))
            xk_1 = tf.concat(xk_1, axis=1)  # (batch, D, 1, Hk_1)
            xk_1 = tf.squeeze(xk_1, 2)  # (batch, D, Hk_1)
            xk_1 = tf.transpose(xk_1, [0, 2, 1])  # (batch, Hk_1, D)
            xk_1_pool = tf.reduce_sum(xk_1, 2)
            cin_output.append(xk_1_pool)
            xk = xk_1
        cin_output = tf.concat(cin_output, axis=1)
        return cin_output

    def get_config(self):
        config = super(CIN, self).get_config().copy()
        config['D'] = self.D
        config['Hk'] = self.Hk
        config['k'] = self.k
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class xDeepFM(tf.keras.Model):
    def __init__(self, category_size, numerical_size, feature_size, embedding_size, Hk=10, k=5, **kwargs):
        super(xDeepFM, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_size, name='embedding_layer')
        self.linear_layer = tf.keras.layers.Embedding(feature_size, 1, name='linear_layer')
        self.cin_layer = CIN(embedding_size, Hk, k)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense_layer2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense_layer3 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        # Embedding
        embedding_cate = self.embedding_layer(inputs['category_index'])
        embedding_nume = self.embedding_layer(inputs['numerical_index'])
        embedding_nume = tf.multiply(embedding_nume, tf.expand_dims(inputs['numerical_value'], -1))
        embeddings = tf.concat([embedding_cate, embedding_nume], axis=1)
        # Compressed Interaction Network
        cin_output = self.cin_layer(embeddings)
        # Liner
        linear_cate = self.linear_layer(inputs['category_index'])
        linear_nume = self.linear_layer(inputs['numerical_index'])
        linear_nume = tf.multiply(linear_nume, tf.expand_dims(inputs['numerical_value'], 2))
        linear_output = tf.concat([linear_cate, linear_nume], axis=1)
        linear_output = tf.squeeze(linear_output, 2)
        # DNN
        dense_output = self.flatten_layer(embeddings)
        dense_output = self.dense_layer1(dense_output)
        dense_output = self.dense_layer2(dense_output)
        dense_output = self.dense_layer3(dense_output)
        # Output
        output = self.concat_layer([cin_output, linear_output, dense_output])
        output = self.output_layer(output)
        return output


if __name__ == '__main__':
    # model parameters
    params = dict()
    params['category_size'] = 12
    params['numerical_size'] = 10
    params['feature_size'] = 5000
    params['embedding_size'] = 8
    # generate test data
    batch_size = 16
    data = dict()
    data['category_index'] = np.random.randint(0, 5000, (batch_size, params['category_size']), np.int32)
    data['numerical_index'] = np.random.randint(0, 5000, (batch_size, params['numerical_size']), np.int32)
    data['numerical_value'] = np.random.randn(batch_size, params['numerical_size']).astype(np.float32)
    # build model
    model = xDeepFM(**params)
    result = model(data)

