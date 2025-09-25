import numpy as np
import torch
from torch import Tensor

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def calc_accuracy(preds: Tensor, labels: Tensor):
    if preds.shape[-1]==1:
        accuracy = ((preds[:,0] >= 0.5).float() == labels).float().mean() * 100
    else:
        predicted_labels = torch.argmax(preds, dim=1)
        correct = (predicted_labels == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
    return accuracy


def get_results_classifier_sklearn(
        x_train: Tensor, 
        y_train: Tensor,
        x_test: Tensor, 
        y_test: Tensor,
        n_layers: int=2,
        max_iter=50,
        **kwargs
    ):
    x_train, y_train = x_train.numpy(), y_train.numpy()
    x_test, y_test = x_test.numpy(), y_test.numpy()
    n_classes = int(y_train.max()+1)
    if n_layers>0: 
        x_dim, y_dim = x_train.shape[-1], int(y_train.max()+1)
        hidden_sizes = tuple(np.linspace(x_dim, y_dim, n_layers+2, dtype=int)[1:-1])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_sizes, 
            activation='relu', 
            solver='adam', 
            max_iter=max_iter, 
            batch_size=256,
            random_state=42
        )
    else:
        model = LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=max_iter
        )
    model.fit(x_train, y_train)
    class_pred = model.predict(x_test)
    y_pred = model.predict_proba(x_test)
    y_pred[y_pred == 0] = 1e-6 
    probs_class = np.bincount(y_train.astype(int), minlength=n_classes) / len(y_train)
    entropy = -(probs_class * np.log(probs_class)).sum()
    cond_entropy = -(y_pred * np.log(y_pred)).sum(axis=1).mean()
    del model
    return {
        'accuracy': accuracy_score(y_test, class_pred) * 100,
        'mi': 1 - cond_entropy / entropy,
    }


def calc_ece(preds: Tensor, labels: Tensor, n_bins: int = 15):
    confidences, y_hat = preds.max(dim=1)
    accuracies = (y_hat == labels).float()

    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=preds.device)
    ece = torch.tensor(0.0, device=preds.device)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            prop = in_bin.float().mean()
            ece += (confidences[in_bin].mean() - accuracies[in_bin].mean()).abs() * prop

    return float(ece.item())


def calc_brier(preds, labels) -> float:
    """
    Multiclass Brier score without building one-hot.
    For each sample: sum_k (p_k - y_k)^2 = 1 - 2*p_true + sum_k p_k^2
    """
    p_true = preds.gather(1, labels.view(-1, 1)).squeeze(1)
    sum_p2 = (preds ** 2).sum(dim=1)
    brier = (1.0 - 2.0 * p_true + sum_p2).mean()
    return float(brier.item())



def calc_map(preds, labels) -> float:
    # Flatten
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Sort by predicted score
    sorted_indices = torch.argsort(preds, descending=True)
    sorted_labels = labels[sorted_indices]

    # Cumulative true positives and false positives
    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1 - sorted_labels, dim=0)

    precisions = tp / (tp + fp)
    recalls = tp / labels.sum()

    # If no positives, AP is undefined → return 0
    if labels.sum() == 0:
        return 0.0

    # Interpolated AP = sum over recall steps
    AP = torch.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    return AP.item()
