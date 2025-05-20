import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from .model_utils import RandomModel

def plot_hist(history, oldhist=None):
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(history["accuracy"], label="train", color="green")
    plt.plot(history["val_accuracy"], label="val", color="red")
    plt.title("Accuracy")
    plt.legend()

    if oldhist:
        plt.subplot(122)
        plt.plot(oldhist["accuracy"], label="old_train", color="cyan")
        plt.plot(oldhist["val_accuracy"], label="old_val", color="pink")
        plt.title("Old Accuracy")
        plt.legend()
    plt.show()

def display_confusion_matrix(true, pred, class_names):
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(30, 30))
    disp.plot(ax=ax)
    plt.xticks(rotation=45)
    plt.show()

def display_classification_report(true, pred, class_names):
    report = classification_report(true, pred, target_names=class_names)
    print(report)
    return report

def get_true_values_with_predictions(test_set, model=None):
    predictions, predictions_proba, true_values = [], [], []

    if model is None:
        print("WARNING: No model provided, generating random predictions.")
        model = RandomModel(len(test_set.class_names))

    for x, y in test_set:
        batch_preds = model.predict(x, verbose=0)
        predictions.extend(np.argmax(batch_preds, axis=1))
        predictions_proba.extend(np.max(batch_preds, axis=1))
        true_values.extend(y.numpy())

    return pd.DataFrame({
        'prediction': predictions,
        'prediction_proba': predictions_proba,
        'true_value': true_values
    })
