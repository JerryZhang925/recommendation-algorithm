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


def build_model(**kwargs):
    # Input Layer
    input_cate_inx = tf.keras.layers.Input(shape=kwargs['category_size'], dtype=tf.int32, name='category_index')
    input_nume_inx = tf.keras.layers.Input(shape=kwargs['numerical_size'], dtype=tf.int32, name='numerical_index')
    input_nume_val = tf.keras.layers.Input(shape=kwargs['numerical_size'], dtype=tf.float32, name='numerical_value')
    # Embedding
    embedding_layer = tf.keras.layers.Embedding(kwargs['feature_size'], kwargs['embedding_size'], name='embedding_layer')
    linear_layer = tf.keras.layers.Embedding(kwargs['feature_size'], 1, name='linear_layer')
    # First Order Part
    first_order_cate = linear_layer(input_cate_inx)
    first_order_nume = linear_layer(input_nume_inx)
    first_order_nume = tf.multiply(first_order_nume, tf.expand_dims(input_nume_val, 2))
    first_order_concat = tf.concat([first_order_cate, first_order_nume], axis=1)
    first_order = tf.reduce_sum(first_order_concat, axis=1)
    # Second Order Part
    embedding_cate = embedding_layer(input_cate_inx)
    embedding_nume = embedding_layer(input_nume_inx)
    embedding_nume = tf.multiply(embedding_nume, tf.expand_dims(input_nume_val, -1))
    embeddings = tf.concat([embedding_cate, embedding_nume], axis=1)
    second_order = BiInteraction(name='second_order')(embeddings)
    second_order = tf.reduce_sum(second_order, axis=1, keepdims=True)
    # Deep Part
    deep_layer = tf.keras.layers.Reshape((embeddings.shape[1]*embeddings.shape[2],))(embeddings)
    deep_layer = tf.keras.layers.Dropout(0.5)(deep_layer)
    deep_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)(deep_layer)
    deep_layer = tf.keras.layers.Dropout(0.5)(deep_layer)
    deep_layer = tf.keras.layers.Dense(64, activation=tf.nn.relu)(deep_layer)
    deep_layer = tf.keras.layers.Dropout(0.5)(deep_layer)
    deep_layer = tf.keras.layers.Dense(32, activation=tf.nn.relu)(deep_layer)
    deep_layer = tf.keras.layers.Dropout(0.5)(deep_layer)
    deep_layer = tf.keras.layers.Dense(1)(deep_layer)
    # Output
    add_layer = tf.keras.layers.Add()([first_order, second_order, deep_layer])
    output = tf.keras.layers.Activation(tf.nn.relu, name='output')(add_layer)
    # Build Model
    model = tf.keras.Model(inputs=[input_cate_inx, input_nume_inx, input_nume_val], outputs=output)
    model.compile(optimizer=kwargs['optimizer'], loss=kwargs['loss'], metrics=kwargs['metrics'])
    return model


if __name__ == '__main__':
    kwargs = dict()
    kwargs['category_size'] = 12
    kwargs['numerical_size'] = 10
    kwargs['feature_size'] = 5000
    kwargs['embedding_size'] = 8
    kwargs['optimizer'] = tf.keras.optimizers.Adam()
    kwargs['loss'] = tf.keras.losses.BinaryCrossentropy()
    kwargs['metrics'] = [tf.keras.metrics.AUC()]
    model = build_model(**kwargs)
