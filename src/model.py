import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D, Concatenate, Dropout, BatchNormalization, Activation, ZeroPadding3D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.saving import register_keras_serializable
from config import PARTICLE_WEIGHTS, PARTICLE_COLORS, CLASS_MAPPING


# On suppose ici que le nombre total de classes (background + protéines) est 7.
n_classes = 7

def protein_accuracy(y_true, y_pred):
    y_true_labels = tf.argmax(y_true, axis=-1)
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    mask = tf.not_equal(y_true_labels, 0)
    correct = tf.equal(y_true_labels, y_pred_labels)
    correct_masked = tf.boolean_mask(correct, mask)

    return tf.reduce_sum(tf.cast(correct_masked, tf.float32)) / tf.maximum(1.0, tf.reduce_sum(tf.cast(mask, tf.float32)))

@register_keras_serializable()
def weighted_categorical_crossentropy(weights_dict, class_mapping, num_classes):
    """
    Fonction de perte pondérée qui ignore le fond (classe 0).
    """
    weight_vector = np.ones(num_classes, dtype=np.float32)
    for molecule, class_idx in class_mapping.items():
        if molecule in weights_dict:
            weight_vector[class_idx] = weights_dict[molecule]
    weight_vector = tf.constant(weight_vector, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        loss_unweighted = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)

        # Exclure les voxels de classe 0 (fond)
        true_class = tf.argmax(y_true, axis=-1)
        mask = tf.not_equal(true_class, 0)
        voxel_weights = tf.gather(weight_vector, true_class)
        weighted_loss = loss_unweighted * voxel_weights

        # Appliquer le masque : ne considérer que les protéines
        masked_loss = tf.boolean_mask(weighted_loss, mask)

        return tf.cond(
            tf.greater(tf.size(masked_loss), 0),  
            lambda: tf.reduce_mean(masked_loss),  
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    return loss



def conv_block(inputs, n_filters, dropout=0, batch_norm=True):
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

def unet3d_model(input_shape, n_classes, filters=[16, 32, 64], dropout=0):
    inputs = Input(input_shape)
    # Encoder
    c1 = conv_block(inputs, filters[0], dropout)
    p1 = MaxPooling3D(pool_size=(2,2,2), padding="same")(c1)
    
    c2 = conv_block(p1, filters[1], dropout)
    p2 = MaxPooling3D(pool_size=(2,2,2), padding="same")(c2)
    
    c3 = conv_block(p2, filters[2], dropout)
    p3 = MaxPooling3D(pool_size=(2,2,2), padding="same")(c3)
    
    # Bottleneck
    c4 = conv_block(p3, filters[2]*2, dropout)
    # Decoder
    u3 = UpSampling3D(size=(2,2,2))(c4)
    # Ici, c3 a été obtenu avec padding="same". On s'attend à ce que u3 et c3 aient les mêmes dimensions.
    u3 = Concatenate()([u3, c3])
    c5 = conv_block(u3, filters[2], dropout)
    
    u2 = UpSampling3D(size=(2,2,2))(c5)
    u2 = Concatenate()([u2, c2])
    c6 = conv_block(u2, filters[1], dropout)
    
    u1 = UpSampling3D(size=(2,2,2))(c6)
    # On remarque que la dimension spatiale de u1 est calculée par multiplication par 2,
    # ce qui peut donner une taille légèrement différente de celle de c1.
    # Par exemple, avec une entrée de (23,79,79), u1 pourrait avoir (24,80,80).
    # On applique alors un ZeroPadding3D sur c1 pour l'ajuster.
    c1_pad = ZeroPadding3D(padding=((0,1), (0,1), (0,1)))(c1)
    u1 = Concatenate()([u1, c1_pad])
    c7 = conv_block(u1, filters[0], dropout)


    # Cropping pour ajuster la taille finale
    output_crop = Cropping3D(cropping=((0,1), (0,1), (0,1)))(c7)
    outputs = Conv3D(n_classes, kernel_size=1, activation='softmax')(output_crop)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Paramètres d'entrée adaptés à votre cas (par exemple, avec des patches de résolution 2)
input_shape = (23, 79, 79, 1)
n_classes = 7
filters = [16, 32, 64]
dropout = 0



