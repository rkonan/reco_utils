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
from tqdm import tqdm
import numpy as np
import pandas as pd

def get_true_values_with_predictions(test_set, model=None, model_embedding=None):
    predictions, predictions_proba, true_values = [], [], []
    proba_vectors, embeddings, sample_ids = [], [], []

    if model is None:
        raise ValueError("Tu dois fournir un modèle entraîné (ou une classe RandomModel).")

    for i, (x, y) in enumerate(tqdm(test_set, desc="Predicting (Keras)", unit="batch")):
        batch_preds = model.predict(x, verbose=0)  # (batch_size, num_classes)
        batch_true = y.numpy()

        predictions.extend(np.argmax(batch_preds, axis=1))
        predictions_proba.extend(np.max(batch_preds, axis=1))
        proba_vectors.extend(batch_preds)
        true_values.extend(batch_true)
        sample_ids.extend([f"sample_{i}_{j}" for j in range(len(batch_true))])

        # Si un modèle d’embedding est fourni (ex : Model(inputs, penultimate_layer))
        if model_embedding is not None:
            batch_embeddings = model_embedding.predict(x, verbose=0)
            embeddings.extend(batch_embeddings)

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "prediction": predictions,
        "prediction_proba": predictions_proba,
        "true_value": true_values,
        "proba_vector": proba_vectors
    })

    if model_embedding is not None:
        df["embedding"] = embeddings

    return df
@torch.no_grad()
def get_true_values_with_predictions_torch(test_loader, model, device='cuda', use_embeddings=True):
    model.eval()
    model.to(device)

    all_preds = []
    all_pred_probas = []
    all_true = []
    all_pred_vectors = []
    all_embeddings = []
    sample_ids = []

    for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
        inputs = inputs.to(device)
        outputs = model(inputs)

        probas = torch.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probas, dim=1)
        confidence_scores = torch.max(probas, dim=1).values

        all_preds.extend(predicted_classes.cpu().numpy())
        all_pred_probas.extend(confidence_scores.cpu().numpy())
        all_pred_vectors.extend(probas.cpu().numpy())
        all_true.extend(labels.cpu().numpy())
        sample_ids.extend([f"sample_{batch_idx}_{i}" for i in range(len(labels))])

        # Si le modèle a un forward_features (par ex. Swin, ConvNeXt)
        if use_embeddings:
            if hasattr(model, "forward_features"):
                embeddings = model.forward_features(inputs)
                if isinstance(embeddings, tuple):  # certains modèles renvoient des tuples
                    embeddings = embeddings[0]
                all_embeddings.extend(embeddings.cpu().numpy())
            else:
                raise AttributeError("Le modèle ne possède pas de méthode `forward_features()`.")

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "prediction": all_preds,
        "prediction_proba": all_pred_probas,
        "true_value": all_true,
        "proba_vector": all_pred_vectors
    })

    if use_embeddings:
        df["embedding"] = all_embeddings

    return df
