import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    D, H = Wz.shape
    
    # Convert to 2D if need
    x, squeeze_x = _as2d(x, D)
    h_prev, squeeze_h = _as2d(h_prev, H)

    # Update gate
    z = _sigmoid(x @ Wz + h_prev @ Uz + bz)

    # Reset Gate
    r = _sigmoid(x @ Wr + h_prev @ Ur + br)

    # Candidate hidden state
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)

    # final hidden state
    h_next = (1 - z) * h_prev + z * h_tilde

    # Restore original shape if input was 1D
    if squeeze_x and squeeze_h:
        h_next = h_next.reshape(-1)
    
    return h_next