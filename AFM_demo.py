import tensorflow as tf
import numpy as np


class PairWise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PairWise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pair_wise_index = [[i, j] for i in range(input_shape[1] - 1) for j in range(i + 1, input_shape[1])]
        # self.pair_wise_index = tf.constant(pair_wise_index, dtype=tf.int32)
        super(PairWise, self).build(input_shape)

    def call(self, inputs):
        pair_product = tf.transpose(inputs, [1, 0, 2])
        pair_product = tf.nn.embedding_lookup(pair_product, self.pair_wise_index)
        pair_product = tf.reduce_prod(pair_product, axis=1)
        pair_product = tf.transpose(pair_product, [1, 0, 2])
        return pair_product

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int((input_shape[1] - 1) * input_shape[1] / 2), input_shape[2])

    def get_config(self):
        config = super().get_config().copy()
        config.update({'pair_wise_index': self.pair_wise_index})
        return config


class Attention(tf.keras.layers.Layer):
    def __init__(self, hidden_size=32, **kwargs):
        self.hidden_size = hidden_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.hidden_size), dtype=tf.float32,
                                 initializer='glorot_normal')
        self.b = self.add_weight(name='b', shape=(self.hidden_size,), dtype=tf.float32, initializer='zeros')
        self.h = self.add_weight(name='h', shape=(self.hidden_size, 1), dtype=tf.float32, initializer='glorot_normal')
        super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, inputs, **kwargs):
        attention_value = tf.nn.relu(tf.add(tf.matmul(inputs, self.W), self.b))
        attention_value = tf.matmul(attention_value, self.h)
        attention_value = tf.nn.softmax(attention_value, axis=1)
        attention_embedding = tf.multiply(attention_value, inputs)
        attention_embedding = tf.reduce_sum(attention_embedding, axis=1)
        return attention_embedding

    def get_config(self):
        config = super().get_config().copy()
        config.update({'hidden_size': self.hidden_size})
        return config


class AFM(tf.keras.Model):
    def __init__(self, feature_size, embedding_size, *args, **kwargs):
        super(AFM, self).__init__(*args, **kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(feature_size, embedding_size)
        self.linear_layer = tf.keras.layers.Embedding(feature_size, 1)
        self.pair_wise_layer = PairWise()
        self.attention_layer = Attention()
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def call(self, inputs, training=None, mask=None):
        # Shallow Part
        first_order_cate = self.linear_layer(inputs['category_index'])
        first_order_nume = self.linear_layer(inputs['numerical_index'])
        first_order_nume = tf.multiply(first_order_nume, tf.expand_dims(inputs['numerical_value'], -1))
        first_order_concat = tf.concat([first_order_cate, first_order_nume], axis=1)
        first_order = tf.reduce_sum(first_order_concat, axis=1)
        # Embedding
        embedding_cate = self.embedding_layer(inputs['category_index'])
        embedding_nume = self.embedding_layer(inputs['numerical_index'])
        embedding_nume = tf.multiply(embedding_nume, tf.expand_dims(inputs['numerical_value'], -1))
        embeddings = tf.concat([embedding_cate, embedding_nume], axis=1)
        # Attention FM
        pair_wise = self.pair_wise_layer(embeddings)
        attention_embedding = self.attention_layer(pair_wise)
        attention_output = tf.reduce_sum(attention_embedding, axis=1, keepdims=True)
        # Output
        output = tf.nn.sigmoid(tf.add(first_order, attention_output))
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
    model = AFM(**params)
    result = model(data)