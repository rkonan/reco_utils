import tensorflow as tf
from .constants import *
import os

def get_img_size(model_type):
    sizes = {
        "EfficientNetV2M": IMG_SIZE_EfficientNetV2M,
        "ResNet50V2": IMG_SIZE_ResNet50V2,
        "ResNet101V2": IMG_SIZE_ResNet101V2,
        "ViT": IMG_SIZE_ViT,
        "ConvNeXt": IMG_SIZE_ConvNeXt,
        "Efficientnetb3": IMG_SIZE_EfficientNetB3
    }
    return sizes.get(model_type, 224)

def load_dataset(model_type,train_path, test_path, batch_size=BATCH_SIZE):
    img_size = get_img_size(model_type)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path, validation_split=0.15, subset="training", seed=SEED,
        image_size=(img_size, img_size), batch_size=batch_size, label_mode="int"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path, validation_split=0.15, subset="validation", seed=SEED,
        image_size=(img_size, img_size), batch_size=batch_size, label_mode="int"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path, image_size=(img_size, img_size), label_mode="int"
    )

    return train_ds, val_ds, test_ds

def targeted_batch_augment(images, labels, target_ids=None):
    if target_ids is None:
        target_ids = []

    target_ids = tf.constant(target_ids, dtype=labels.dtype)
    apply_to_all = tf.equal(tf.size(target_ids), 0)

    def apply_conditionally(img, label):
        def augment():
            img_aug = tf.image.random_flip_left_right(img)
            img_aug = tf.image.random_brightness(img_aug, max_delta=0.2)
            img_aug = tf.image.random_contrast(img_aug, 0.8, 1.2)
            img_aug = tf.image.random_saturation(img_aug, 0.8, 1.2)
            img_aug = tf.image.random_hue(img_aug, 0.05)
            img_aug = tf.image.random_crop(img_aug,size=(224,224,3))
            return img_aug

        if isinstance(apply_to_all, tf.Tensor):
            condition = tf.cond(
                apply_to_all,
                lambda: True,
                lambda: tf.reduce_any(tf.equal(label, target_ids))
            )
        else:
            condition = apply_to_all or tf.reduce_any(tf.equal(label, target_ids))

        return tf.cond(condition, augment, lambda: img)

    augmented_images = tf.map_fn(
        lambda x: apply_conditionally(x[0], x[1]),
        (images, labels),
        fn_output_signature=tf.TensorSpec(shape=images.shape[1:], dtype=images.dtype)
    )

    return augmented_images, labels

#========= pour créer des mock datatset de tests 
def create_mock_dataset(name =None,num_samples=100, img_size=224, num_classes=5):
    if name is not  None:
        img_size=get_img_size(name)
    x = tf.random.uniform((num_samples, img_size, img_size, 3))
    y = tf.random.uniform((num_samples,), minval=0, maxval=num_classes, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return ds.batch(8)
