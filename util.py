import numpy as np
from sklearn.metrics import confusion_matrix

def matrix_confusao(y, preds):
    """
    INPUT:
        y = The correct Target values
    OUTPUT:
        A array with:
        True Positive, False Negative
        False Positive, True Negative       
    """
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    return np.array([[tp, fp],[fn, tn]])