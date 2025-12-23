from sklearn.metrics import precision_score, recall_score, hamming_loss
import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score, roc_auc_score

def calculate_custom_metrics(true_labels, predicted_labels, max_tags=4):
    true_labels = true_labels
    predicted_labels = predicted_labels

    metrics = []


    true_counts = np.sum(true_labels == 1, axis=1)

    for j in range(max_tags, 0, -1):
        P_j_sum = 0
        C_j_sum = 0

        for k in range(j, max_tags + 1):

            valid_base_samples = true_counts >= k

            match_counts = np.sum((true_labels == 1) & (predicted_labels == 1), axis=1)

            P_k = np.sum((match_counts >= j) & valid_base_samples)

            C_k = np.sum(valid_base_samples)

            P_j_sum += P_k
            C_j_sum += C_k

        MR_j = P_j_sum / C_j_sum if C_j_sum > 0 else 0
        metrics.append(MR_j)

    return metrics


def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """
     Sensitivity (Recall), Specificity, Accuracy, MCC, F1-score, AUC
    """
    y_pred = (y_pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率/灵敏度 (SN)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异度 (SP)
    accuracy = accuracy_score(y_true, y_pred)  # 准确率
    mcc = matthews_corrcoef(y_true, y_pred)  # MCC
    f1 = f1_score(y_true, y_pred)  # F1-score
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None  # AUC (需要概率)

    print(f"{sensitivity:.4f}\t{specificity:.4f}\t{accuracy:.4f}\t{mcc:.4f}\t{f1:.4f}")
    return {
        "Sensitivity (SN)": round(sensitivity, 4),
        "Specificity (SP)": round(specificity, 4),
        "Accuracy": round(accuracy, 4),
        "MCC": round(mcc, 4),
        "F1-score": round(f1, 4),
        "AUC": round(auc, 4) if auc is not None else "N/A"
    }


def multi_label_metrics_binary(Y_true, Y_score, threshold=0.5):

    Y_pred = (Y_score >= threshold).astype(int)
    n, M = Y_true.shape

    aiming = 0.0
    coverage = 0.0
    accuracy = 0.0
    absolute_true = 0.0
    absolute_false = 0.0

    for i in range(n):
        yt = set(np.where(Y_true[i] == 1)[0])
        yp = set(np.where(Y_pred[i] == 1)[0])

        intersection = yt & yp
        union = yt | yp

        aiming += len(intersection) / len(yp) if len(yp) > 0 else 0
        coverage += len(intersection) / len(yt) if len(yt) > 0 else 0
        accuracy += len(intersection) / len(union) if len(union) > 0 else 0
        absolute_true += int(yt == yp)
        absolute_false += (len(union) - len(intersection)) / M

    return {
        "Aiming": aiming / n,
        "Coverage": coverage / n,
        "Accuracy": accuracy / n,
        "Absolute-True": absolute_true / n,
        "Absolute-False": absolute_false / n
    }


