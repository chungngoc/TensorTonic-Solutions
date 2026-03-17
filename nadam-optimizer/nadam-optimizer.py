import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    grad = np.asarray(grad)
    w = np.asarray(w)
    # First moment
    m_new = beta1 * np.asarray(m) + (1-beta1) * grad
    # Second moment
    v_new = beta2 * np.asarray(v) + (1-beta2) * (grad ** 2)

    # Nesterov-Ajusted
    w_new = w - lr* (beta1 * m_new + (1-beta1)*grad) / (np.sqrt(v_new) + eps)
    return (w_new, m_new, v_new)

    