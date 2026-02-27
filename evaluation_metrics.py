import numpy as np

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        i = class_to_index[true]
        j = class_to_index[pred]
        matrix[i, j] += 1

    return matrix


def f1_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    f1_scores = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        f1_scores.append(f1)

    return np.mean(f1_scores)