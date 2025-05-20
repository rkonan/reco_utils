import tensorflow as tf
import time
from keras.callbacks import Callback,ModelCheckpoint
import os
import zipfile
import shutil

#=== Callback pour afficher le learning rate ===
class PrintLR(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        #lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        lr = self.model.optimizer.learning_rate.numpy()
        print(f"Learning rate at epoch {epoch+1}: {lr:.6f}")

#==== call back pour mesurer le temps ====
class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        # Capture l'heure de début
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        # Capture l'heure de fin et calcule la durée
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"Temps d'entraînement: {self.duration:.2f} secondes")


class ZipModelCheckpoint(ModelCheckpoint):
    def __init__(self, model_to_save, file_path, name,mode, **kwargs):
        # On utilise un fichier factice pour désactiver le vrai ModelCheckpoint
        super().__init__(filepath="placeholder.keras", save_weights_only=False, **kwargs)
        self.model_to_save = model_to_save
        self.file_path = file_path
        self.name = name
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        print("ici")
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        # Détermine si la condition de sauvegarde est remplie
        if self._is_improvement(current, self.best):
            print("📦 Sauvegarde du modèle ZIP à l’époque", epoch+1)
            self.best = current
            zip_path = self.file_path.format(epoch=epoch + 1)
            self._save_vit_classifier_zip(self.model_to_save, zip_path)

    def _is_improvement(self, current, best):
        if self.mode == 'min':
            return current < best
        else:
            return current > best

    def _save_vit_classifier_zip(self, model, zip_path):
        temp_dir = "vit_temp_save"
        keras_model_path = os.path.join(temp_dir, "keras_model.keras")
        vit_weights_path = os.path.join(temp_dir, "vit_finetuned_weights.h5")

        os.makedirs(temp_dir, exist_ok=True)
        model.save(keras_model_path)
        model.vit.save_weights(vit_weights_path)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for folder_name, _, filenames in os.walk(temp_dir):
                for filename in filenames:
                    file_path = os.path.join(folder_name, filename)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)

        shutil.rmtree(temp_dir)
        print(f"✅ Modèle sauvegardé dans {zip_path}")