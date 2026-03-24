import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    stride = image_size / feature_size

    anchors = []

    for i in range(feature_size):
        for j in range(feature_size):
            # center of the anchor
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)

                    x_min = cx - w / 2
                    y_min = cy - h / 2
                    x_max = cx + w / 2
                    y_max = cy + h / 2

                    anchors.append([x_min, y_min, x_max, y_max])
    return anchors