import cv2
import numpy as np
from typing import Tuple, Union
from .utils import read_image_rgb


class QCValidator:

    def __init__(self, peak_threshold: float = 0.25, qc_window: Tuple[int, int] = (50, 500)):

        self.peak_threshold = peak_threshold
        self.qc_window = qc_window

    def validate(self, qc_image_input: Union[str, np.ndarray]) -> Tuple[bool, float]:

        if isinstance(qc_image_input, str):
            img_rgb = read_image_rgb(qc_image_input, target_size=(220, 700))
        else:
            img_rgb = qc_image_input

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        inv_gray = 255.0 - gray.astype(np.float32)

        proj = np.mean(inv_gray, axis=1)

        start_y, end_y = self.qc_window
        window_proj = proj[start_y:end_y]

        if len(window_proj) == 0:
            return False, 0.0

        max_signal = np.max(window_proj)
        min_baseline = np.min(proj)
        
        peak_height = (max_signal - min_baseline) / 255.0

        is_valid = bool(peak_height >= self.peak_threshold)

        return is_valid, float(peak_height)