from tensorflow.keras import layers, models

from .constants import *
from data_utils import get_img_size

def get_preprocess_input(model_type):
    if model_type == "EfficientNetV2M":
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        return preprocess_input
    elif model_type in ("ResNet50V2", "ResNet101V2"):
        from tensorflow.keras.applications.resnet_v2 import preprocess_input
        return preprocess_input
    elif model_type == "ViT":
        return lambda x: x
    elif "ConvNeXt" in model_type:
        return lambda x: x / 255.0

def build_model(base, model_type):
    if model_type == "ViT":
        return base

    img_size = get_img_size(model_type)
    base.trainable = False
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs)
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(train_dataset, class_names):
    labels = []
    class_counts = {}
    for idx, class_name in enumerate(class_names):
        count = len(os.listdir(os.path.join(train_dataset, class_name)))
        class_counts[class_name] = count
        labels.extend([idx] * count)

    print("Répartition des classes :", class_counts)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Poids calculés :", weight_dict)
    return weight_dict, class_counts
