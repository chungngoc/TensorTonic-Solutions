def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # Write code here
    sum_ref = sum(reference_counts)
    sum_prod = sum(production_counts)

    # Normalize histograms
    reference_counts = [i / sum_ref for i in reference_counts]
    production_counts = [i / sum_prod for i in production_counts]

    # Compute Total Variation Distance
    tvd = 0
    for (p,q) in zip(reference_counts, production_counts):
        tvd += abs(p-q)
    score = 0.5 * tvd
    drift_detected = score > threshold

    return {"score": score, "drift_detected": drift_detected}