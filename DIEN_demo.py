import tensorflow as tf
import numpy as np


class Dice(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, axis=-1, **kwargs):
        self.epsilon = epsilon
        self.axis = axis
        super(Dice, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', shape=(), dtype=tf.float32, initializer='zeros')
        self.bn = tf.keras.layers.BatchNormalization(axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        super(Dice, self).build(input_shape)

    def call(self, inputs, training, **kwargs):
        norm_data = tf.nn.sigmoid(self.bn(inputs, training=training))
        output = self.alpha * (1.0 - norm_data) * inputs + norm_data * inputs
        return output


class AuxiliaryLoss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AuxiliaryLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layer = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        super(AuxiliaryLoss, self).build(input_shape)

    def call(self, interest, pos_target, neg_target, **kwargs):
        interest_repeat = tf.repeat(tf.expand_dims(interest, 1), neg_target.get_shape()[1], axis=1)
        neg_loss = self.dense_layer(tf.concat([interest_repeat, neg_target], -1))
        pos_loss = self.dense_layer(tf.concat([interest, pos_target], -1))
        neg_loss = -tf.math.log(1 - neg_loss)
        pos_loss = -tf.expand_dims(tf.math.log(pos_loss), axis=1)
        aux_loss = tf.reduce_mean(tf.concat([neg_loss, pos_loss], axis=1))
        return aux_loss


class AIGRU(tf.keras.layers.Layer):
    def __init__(self, gru_ndim, target_ndim, **kwargs):
        self.gru_layer = tf.keras.layers.GRU(gru_ndim, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(target_ndim, use_bias=False)
        super(AIGRU, self).__init__(**kwargs)

    def call(self, key, values, **kwargs):
        gru_state = self.gru_layer(values)
        gru_state = self.dense_layer(gru_state)
        key = tf.expand_dims(key, 1)
        attention_score = tf.matmul(gru_state, key, transpose_b=True)
        attention_score = tf.nn.softmax(attention_score, axis=1)
        output = tf.multiply(gru_state, attention_score)
        output = tf.reduce_sum(output, axis=1)
        return output


class DIEN(tf.keras.Model):
    def __init__(self, field_size, sequence_size, user_feature_size, cont_feature_size,
                 embedding_ndim, embedding_size, negative_num, *args, **kwargs):
        super(DIEN, self).__init__(*args, **kwargs)
        self.field_size = field_size
        self.sequence_size = sequence_size
        self.user_feature_size = user_feature_size
        self.cont_feature_size = cont_feature_size
        self.embedding_ndim = embedding_ndim
        self.embedding_size = embedding_size
        self.negative_num = negative_num
        # Embedding Layer
        self.embedding_layer = tf.keras.layers.Embedding(embedding_size, embedding_ndim)
        # Interest Extractor Layer
        self.gru_layer = tf.keras.layers.GRU(field_size * embedding_ndim, return_sequences=True)
        # Auxiliary Loss
        self.aux_loss_layer = AuxiliaryLoss()
        # Interest Evolution Layer
        self.aigru_layer = AIGRU(embedding_ndim, field_size * embedding_ndim)
        # Deep Layer
        self.dense_layer_1 = tf.keras.layers.Dense(200)
        self.activate_layer_1 = Dice()
        self.dense_layer_2 = tf.keras.layers.Dense(80)
        self.activate_layer_2 = Dice()
        self.output_layer = tf.keras.layers.Dense(2, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        # Embedding
        user_embeddings = self.embedding_layer(inputs['user_features'])
        user_embeddings = tf.reshape(user_embeddings, [-1, self.user_feature_size * self.embedding_ndim])
        context_embeddings = self.embedding_layer(inputs['context_features'])
        context_embeddings = tf.reshape(context_embeddings, [-1, self.cont_feature_size * self.embedding_ndim])
        candidate_embeddings = self.embedding_layer(inputs['candidate_ids'])  # (batch, field, dimmension)
        candidate_embeddings = tf.reshape(candidate_embeddings, [-1, self.field_size * self.embedding_ndim])
        negative_embeddings = self.embedding_layer(inputs['negative_ids'])  # (batch, negative, sequence-1, field, dimmension)
        neg_shape = [-1, self.negative_num, self.sequence_size - 1, self.field_size * self.embedding_ndim]
        negative_embeddings = tf.reshape(negative_embeddings, neg_shape)
        behavior_embeddings = self.embedding_layer(inputs['behavior_ids'])  # (batch, sequence, field, dimmension)
        beh_shape = [-1, self.sequence_size, self.field_size * self.embedding_ndim]
        behavior_embeddings = tf.reshape(behavior_embeddings, beh_shape)
        # Interest Extractor Layer
        gru_interest = self.gru_layer(behavior_embeddings)
        # Auxiliary Loss
        aux_loss = self.aux_loss_layer(gru_interest[:, :-1, :], behavior_embeddings[:, 1:, :], negative_embeddings)
        # Interest Evolution Layer
        aigru_interest = self.aigru_layer(candidate_embeddings, gru_interest)
        # Concat
        embeddings = tf.concat([user_embeddings, context_embeddings, aigru_interest, candidate_embeddings], axis=1)
        dense_vector = self.dense_layer_1(embeddings)
        dense_vector = self.activate_layer_1(dense_vector, training)
        dense_vector = self.dense_layer_2(dense_vector)
        dense_vector = self.activate_layer_2(dense_vector, training)
        output = self.output_layer(dense_vector)
        return output, aux_loss

if __name__ == '__main__':
    batch_size = 16
    # model parameters
    params = dict()
    params['field_size'] = 3
    params['sequence_size'] = 10
    params['user_feature_size'] = 8
    params['cont_feature_size'] = 5
    params['embedding_ndim'] = 8
    params['embedding_size'] = 5000
    params['negative_num'] = 20
    # generate test data
    data = dict()
    data['candidate_ids'] = np.random.randint(0, 5000, (batch_size, params['field_size']), np.int32)
    data['negative_ids'] = np.random.randint(0, 5000, (batch_size, params['negative_num'], params['sequence_size'] - 1,
                                               params['field_size']), np.int32)
    data['behavior_ids'] = np.random.randint(0, 5000, (batch_size, params['sequence_size'], params['field_size']),
                                             np.int32)
    data['user_features'] = np.random.randint(0, 5000, (batch_size, params['user_feature_size']), np.int32)
    data['context_features'] = np.random.randint(0, 5000, (batch_size, params['cont_feature_size']), np.int32)
    data['sequence_num'] = np.random.randint(1, params['sequence_size'], (batch_size, params['field_size']), np.int32)
    # build model
    model = DIEN(**params)
    result, aux_loss = model(data)




