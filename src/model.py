import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv3D, MaxPooling3D, UpSampling3D, 
                                     Cropping3D, Concatenate, Dropout, BatchNormalization, 
                                     Activation, ZeroPadding3D)
from tensorflow.keras.saving import register_keras_serializable
from config import PARTICLE_WEIGHTS, PARTICLE_COLORS, CLASS_MAPPING
from tensorflow.keras.optimizers import Adam

@register_keras_serializable()
def weighted_categorical_crossentropy(weights_dict, class_mapping, num_classes):
    weight_vector = np.ones(num_classes, dtype=np.float32)
    for molecule, class_idx in class_mapping.items():
        if molecule in weights_dict:
            weight_vector[class_idx] = weights_dict[molecule]
    weight_vector = tf.constant(weight_vector, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        loss_unweighted = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        true_class = tf.argmax(y_true, axis=-1)
        mask = tf.not_equal(true_class, 0)
        voxel_weights = tf.gather(weight_vector, true_class)
        weighted_loss = loss_unweighted * voxel_weights
        masked_loss = tf.boolean_mask(weighted_loss, mask)

        return tf.cond(tf.size(masked_loss) > 0, 
                       lambda: tf.reduce_mean(masked_loss), 
                       lambda: tf.constant(0.0, dtype=tf.float32))
    return loss

class UNet3D(Model):
    def __init__(self, input_shape=(23, 79, 79, 1), n_classes=7, filters=[16, 32, 64], dropout=0):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.filters = filters
        self.dropout = dropout
        self.model = self.build_model()

    def conv_block(self, inputs, n_filters, dropout, batch_norm=True):
        x = Conv3D(n_filters, kernel_size=3, padding='same')(inputs)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(n_filters, kernel_size=3, padding='same')(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout > 0:
            x = Dropout(dropout)(x)
        return x

    def build_model(self):
        inputs = Input(self.input_shape)
        c1 = self.conv_block(inputs, self.filters[0], self.dropout)
        p1 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(c1)

        c2 = self.conv_block(p1, self.filters[1], self.dropout)
        p2 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(c2)

        c3 = self.conv_block(p2, self.filters[2], self.dropout)
        p3 = MaxPooling3D(pool_size=(2, 2, 2), padding="same")(c3)

        c4 = self.conv_block(p3, self.filters[2] * 2, self.dropout)

        u3 = UpSampling3D(size=(2, 2, 2))(c4)
        u3 = Concatenate()([u3, c3])
        c5 = self.conv_block(u3, self.filters[2], self.dropout)

        u2 = UpSampling3D(size=(2, 2, 2))(c5)
        u2 = Concatenate()([u2, c2])
        c6 = self.conv_block(u2, self.filters[1], self.dropout)

        u1 = UpSampling3D(size=(2, 2, 2))(c6)
        c1_pad = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(c1)
        u1 = Concatenate()([u1, c1_pad])
        c7 = self.conv_block(u1, self.filters[0], self.dropout)

        output_crop = Cropping3D(cropping=((0, 1), (0, 1), (0, 1)))(c7)
        outputs = Conv3D(self.n_classes, kernel_size=1, activation='softmax')(output_crop)

        return Model(inputs=inputs, outputs=outputs)

    def call(self, inputs):
        return self.model(inputs)

    def compile_model(self, learning_rate=0.001):
        optimizer = Adam(learning_rate)
        loss = weighted_categorical_crossentropy(PARTICLE_WEIGHTS, CLASS_MAPPING, self.n_classes)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[self.protein_accuracy])

    @staticmethod
    def protein_accuracy(y_true, y_pred):
        y_true_labels = tf.argmax(y_true, axis=-1)
        y_pred_labels = tf.argmax(y_pred, axis=-1)
        mask = tf.not_equal(y_true_labels, 0)
        correct = tf.equal(y_true_labels, y_pred_labels)
        correct_masked = tf.boolean_mask(correct, mask)
        return tf.reduce_sum(tf.cast(correct_masked, tf.float32)) / tf.maximum(1.0, tf.reduce_sum(tf.cast(mask, tf.float32)))
