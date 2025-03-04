import tensorflow as tf
import numpy as np
from config import CLASS_MAPPING
from src.preprocessing import generate_mask, extract_cubic_patches
from tensorflow.keras.utils import to_categorical

class TomogramDataLoader:
    def __init__(self, dataset, resolution='2', dim_in= 64, sphere_radius=2, augment=False):
        self.dataset = dataset
        self.resolution = resolution
        self.sphere_radius = sphere_radius
        self.augment = augment
        self.dim_in = dim_in

    def generate_patch(self, tomogram):
        mask = generate_mask(tomogram, self.resolution, self.sphere_radius)
        patches_img, patches_mask = extract_cubic_patches(tomogram, self.resolution, mask, dim_in=self.dim_in)
        
        for img, mask in zip(patches_img, patches_mask):
            img = img[..., np.newaxis].astype(np.float32)  # Ajouter la dimension canal
            mask = to_categorical(mask, num_classes=len(CLASS_MAPPING)).astype(np.uint8)
            yield img, mask

    def augment_patch(self, img, mask):
        if self.augment:
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_left_right(img)
                mask = tf.image.flip_left_right(mask)
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_up_down(img)
                mask = tf.image.flip_up_down(mask)
        return img, mask

    def balance_patch(self, img, mask):
        contains_protein = tf.reduce_any(tf.greater(mask[..., 1:], 0))  # Classe 0 est le fond
        return contains_protein

    def dataset_generator(self):
        for tomogram in self.dataset:
            for img, mask in self.generate_patch(tomogram):
                yield img, mask

    def get_dataset(self, batch_size=1):
        dataset = tf.data.Dataset.from_generator(
            self.dataset_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None, None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None, None, len(CLASS_MAPPING)), dtype=tf.uint8)
            )
        )
        # Balance le dataset (50% avec protéines, 50% sans protéines)
        dataset = dataset.filter(self.balance_patch)
        dataset = dataset.concatenate(dataset.filter(lambda img, mask: tf.logical_not(self.balance_patch(img, mask))))
        dataset = dataset.shuffle(1000)
        dataset = dataset.map(self.augment_patch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
