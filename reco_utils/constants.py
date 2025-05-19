from tensorflow.keras import mixed_precision

# Mixed precision
mixed_precision.set_global_policy("mixed_float16")

# Global parameters
BATCH_SIZE = 32
EPOCHS = 20
EPOCHS_FINE_TUNING = 5
NUM_CLASSES = 55
SEED = 42

# Image sizes
IMG_SIZE_EfficientNetV2M = 480
IMG_SIZE_ResNet50V2 = 224
IMG_SIZE_ResNet101V2 = 224
IMG_SIZE_ViT = 224
IMG_SIZE_ConvNeXt = 224

# Paths
train_dataset = "/.../train"  # Remplace par une variable d’environnement si nécessaire
test_dataset = "/.../test"
