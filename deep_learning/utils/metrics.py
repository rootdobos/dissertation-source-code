import torch
import numpy as np
def quadratic_weighted_kappa(y_true, y_pred, num_classes=6):
    """
    Computes the Quadratic Weighted Kappa (QWK) between true and predicted labels.
    
    Parameters:
        y_true (Tensor): Ground truth labels (shape: [N]).
        y_pred (Tensor): Predicted labels (shape: [N]).
        num_classes (int): Number of classes (default: 6).
    
    Returns:
        float: QWK score
    """
    confusion_matrix = np.zeros((num_classes,num_classes))
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1
    return quadratic_weighted_kappa_cf(confusion_matrix, num_classes)

def quadratic_weighted_kappa_cf(confusion_matrix, num_classes=6):
    """
    Computes the Quadratic Weighted Kappa (QWK) from confusion matrix.
    
    Parameters:
        confusion_matrix: -
        num_classes (int): Number of classes (default: 6).
    
    Returns:
        float: QWK score
    """
    O = torch.from_numpy(confusion_matrix)

    # Normalize O
    O = O / O.sum()
    # Compute the expected matrix E
    true_hist = O.sum(dim=1).view(-1, 1)  # True label distribution
    pred_hist = O.sum(dim=0).view(1, -1)  # Predicted label distribution
    E = true_hist @ pred_hist  # Outer product
    E = E / E.sum()  # Normalize E

    # Compute the quadratic weight matrix W
    W = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # Compute QWK
    numerator = (O * W).sum()
    denominator = (E * W).sum()
    kappa = 1.0 - (numerator / denominator)

    return kappa.item()

def compute_precision_recall_f1(confusion_matrix):
    """
    Computes Precision, Recall, and F1-score from a confusion matrix.
    
    Args:
        conf_matrix (torch.Tensor): 6x6 confusion matrix
    
    Returns:
        tuple: (precision_per_class, recall_per_class, f1_per_class, macro_f1, weighted_f1)
    """
    confusion_matrix=torch.from_numpy(confusion_matrix)

    num_classes = confusion_matrix.shape[0]
    # True Positives (Diagonal)
    TP = torch.diag(confusion_matrix)
    # False Positives (Column sum - TP)
    FP = confusion_matrix.sum(dim=0) - TP
    # False Negatives (Row sum - TP)
    FN = confusion_matrix.sum(dim=1) - TP
    # Precision, Recall, F1 per class
    precision = TP / (TP + FP + 1e-8)  # Avoid division by zero
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Macro-Averaged F1
    macro_f1 = f1.mean()

    # Weighted-Averaged F1
    class_counts = confusion_matrix.sum(dim=1)  # Total samples per class
    total_samples = confusion_matrix.sum()
    weights = class_counts / total_samples
    weighted_f1 = (f1 * weights).sum()

    return precision, recall, f1, macro_f1.item(), weighted_f1.item()