import tensorflow as tf
from config.constants import *
import os

def get_img_size(model_type):
    sizes = {
        "EfficientNetV2M": IMG_SIZE_EfficientNetV2M,
        "ResNet50V2": IMG_SIZE_ResNet50V2,
        "ResNet101V2": IMG_SIZE_ResNet101V2,
        "ViT": IMG_SIZE_ViT,
        "ConvNeXt": IMG_SIZE_ConvNeXt,
    }
    return sizes.get(model_type, 224)

def load_dataset(model_type, batch_size=BATCH_SIZE):
    img_size = get_img_size(model_type)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dataset, validation_split=0.15, subset="training", seed=SEED,
        image_size=(img_size, img_size), batch_size=batch_size, label_mode="int"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dataset, validation_split=0.15, subset="validation", seed=SEED,
        image_size=(img_size, img_size), batch_size=batch_size, label_mode="int"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dataset, image_size=(img_size, img_size), label_mode="int"
    )

    return train_ds, val_ds, test_ds

def targeted_batch_augment(images, labels, cassava_target_ids):
    def apply_conditionally(img, label):
        is_cassava = tf.reduce_any(tf.equal(label, cassava_target_ids))

        def augment():
            img_aug = tf.image.random_flip_left_right(img)
            img_aug = tf.image.random_brightness(img_aug, max_delta=0.2)
            img_aug = tf.image.random_contrast(img_aug, 0.8, 1.2)
            img_aug = tf.image.random_saturation(img_aug, 0.8, 1.2)
            img_aug = tf.image.random_hue(img_aug, 0.05)
            return img_aug

        return tf.cond(is_cassava, augment, lambda: img)

    augmented_images = tf.map_fn(
        lambda x: apply_conditionally(x[0], x[1]),
        (images, labels),
        fn_output_signature=tf.TensorSpec(shape=images.shape[1:], dtype=images.dtype)
    )

    return augmented_images, labels
