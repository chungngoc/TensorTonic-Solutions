import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.array(param)
    grad = np.array(grad)
    
    m_new = beta1 * np.array(m) + (1 - beta1) * grad
    v_new = beta2 * np.array(v) + (1 - beta2) * (grad ** 2)

    print(m_new, v_new)
    # bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return (param_new, m_new, v_new)