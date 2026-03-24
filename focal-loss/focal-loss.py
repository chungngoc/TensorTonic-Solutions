import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    p = np.array(p)
    y = np.array(y)
    
    temp1 = ((1 - p) ** gamma) * y * np.log(np.clip(p, 0, 1))
    temp2 = (p ** gamma) * (1 - y) * np.log(np.clip(1-p, 0, 1))

    fl = -(temp1 + temp2)

    return np.mean(fl)