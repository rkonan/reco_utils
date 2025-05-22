from tensorflow.keras import layers, models
import tensorflow as tf
from keras.saving import register_keras_serializable
from transformers import TFAutoModel, AutoConfig, ViTFeatureExtractor


from .constants import *
from .data_utils import get_img_size

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




# === Modèle Vit sérializable  ===
@register_keras_serializable()
class ViTClassifier(tf.keras.Model):
    def __init__(self, num_classes=5, vit_model_name="google/vit-base-patch16-224", **kwargs):
        super().__init__(**kwargs)
        self.vit_model_name = vit_model_name
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(vit_model_name)
        self.vit = TFAutoModel.from_pretrained(vit_model_name)
        self.vit.trainable = False
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32) / 255.0
        pixel_values = (inputs - self.feature_extractor .image_mean) / self.feature_extractor .image_std
        pixel_values = tf.transpose(pixel_values, perm=[0, 3, 1, 2])
        outputs = self.vit(pixel_values=pixel_values, training=training).last_hidden_state
        x = self.pool(outputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.classifier(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.classifier.units,
            "vit_model_name": self.vit_model_name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RandomModel:
    def __init__(self,num_classes):
        self.num_classes = num_classes
    def predict(self,x,verbose=0):
        pred=np.random.random((len(x),self.num_classes))
        return pred
    
#====== un modke model pour tester 
    def build_mock_model(input_shape=(224, 224, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model