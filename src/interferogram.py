import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy.signal import convolve2d


@dataclass
class InterferogramResult:
    wrapped_phase: np.ndarray
    coherence: np.ndarray
    interferogram: np.ndarray
    amplitude_master: np.ndarray
    amplitude_slave: np.ndarray
    mean_coherence: float
    valid_pixel_ratio: float
    aps_phase: Optional[np.ndarray] = None


class Interferogram:
    def __init__(
        self,
        multi_look_window: Tuple[int, int] = (5, 5),
        coherence_threshold: float = 0.3
    ):
        self.multi_look_window = multi_look_window
        self.coherence_threshold = coherence_threshold

    def compute_interferogram(
        self,
        master: np.ndarray,
        slave: np.ndarray,
        use_gaussian: bool = False
    ) -> InterferogramResult:
        start_time = time.time()

        print("  [干涉计算] 共轭相乘生成干涉图...")
        interferogram = master * np.conj(slave)

        print("  [干涉计算] 提取包裹相位...")
        wrapped_phase = np.angle(interferogram)

        print("  [干涉计算] 计算相干系数 (多视视窗)...")
        coherence = self._compute_coherence(
            master,
            slave,
            interferogram,
            use_gaussian=use_gaussian
        )

        print("  [干涉计算] 计算振幅图...")
        amp_master = np.abs(master)
        amp_slave = np.abs(slave)

        mean_coh = np.mean(coherence)
        valid_ratio = np.sum(coherence > self.coherence_threshold) / coherence.size

        print(f"  [干涉计算] 平均相干系数: {mean_coh:.4f}")
        print(f"  [干涉计算] 有效像素比例 (> {self.coherence_threshold}): {valid_ratio:.2%}")

        elapsed = time.time() - start_time
        print(f"  [干涉计算] 完成! 耗时: {elapsed:.2f}s")

        return InterferogramResult(
            wrapped_phase=wrapped_phase,
            coherence=coherence,
            interferogram=interferogram,
            amplitude_master=amp_master,
            amplitude_slave=amp_slave,
            mean_coherence=mean_coh,
            valid_pixel_ratio=valid_ratio,
            aps_phase=None
        )

    def _compute_coherence(
        self,
        master: np.ndarray,
        slave: np.ndarray,
        interferogram: np.ndarray,
        use_gaussian: bool = False
    ) -> np.ndarray:
        if use_gaussian:
            sigma = max(self.multi_look_window) / 2.355
            numerator = gaussian_filter(np.abs(interferogram), sigma=sigma)
            denom_master = gaussian_filter(np.abs(master) ** 2, sigma=sigma)
            denom_slave = gaussian_filter(np.abs(slave) ** 2, sigma=sigma)
        else:
            win_y, win_x = self.multi_look_window
            size = win_y * win_x

            numerator = uniform_filter(np.abs(interferogram), size=self.multi_look_window) * size

            denom_master = uniform_filter(np.abs(master) ** 2, size=self.multi_look_window) * size
            denom_slave = uniform_filter(np.abs(slave) ** 2, size=self.multi_look_window) * size

        denominator = np.sqrt(denom_master * denom_slave)

        coherence = numerator / (denominator + 1e-10)
        coherence = np.clip(coherence, 0, 1)

        return coherence

    def coherence_mask(
        self,
        phase: np.ndarray,
        coherence: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        if threshold is None:
            threshold = self.coherence_threshold

        masked_phase = phase.copy()
        masked_phase[coherence < threshold] = np.nan

        return masked_phase

    def compute_deformation_3d(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        wavelength: float = 0.056,
        incidence_angle: float = 39.0
    ) -> np.ndarray:
        inc_rad = np.radians(incidence_angle)

        deformation = (wrapped_phase * wavelength) / (4 * np.pi * np.cos(inc_rad))

        deformation[coherence < self.coherence_threshold] = np.nan

        return deformation

    def glacier_drift_analysis(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        pixel_size: float = 10.0
    ) -> dict:
        valid_mask = coherence > self.coherence_threshold
        valid_phase = wrapped_phase[valid_mask]

        phase_gradient_y, phase_gradient_x = np.gradient(wrapped_phase)

        phase_gradient_x = median_filter(phase_gradient_x, size=5)
        phase_gradient_y = median_filter(phase_gradient_y, size=5)

        drift_x = (phase_gradient_x / (2 * np.pi)) * pixel_size
        drift_y = (phase_gradient_y / (2 * np.pi)) * pixel_size

        drift_magnitude = np.sqrt(drift_x ** 2 + drift_y ** 2)
        drift_magnitude[~valid_mask] = np.nan

        drift_direction = np.degrees(np.arctan2(drift_y, drift_x))
        drift_direction[~valid_mask] = np.nan

        return {
            'drift_x': drift_x,
            'drift_y': drift_y,
            'drift_magnitude': drift_magnitude,
            'drift_direction': drift_direction,
            'mean_drift': np.nanmean(drift_magnitude),
            'max_drift': np.nanmax(drift_magnitude),
            'std_drift': np.nanstd(drift_magnitude)
        }

    def multi_look(
        self,
        image: np.ndarray,
        window: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        if window is None:
            window = self.multi_look_window

        win_y, win_x = window
        h, w = image.shape

        new_h = h // win_y
        new_w = w // win_x

        cropped = image[:new_h * win_y, :new_w * win_x]

        if np.iscomplexobj(image):
            real_ml = uniform_filter(np.real(cropped), size=window)[::win_y, ::win_x]
            imag_ml = uniform_filter(np.imag(cropped), size=window)[::win_y, ::win_x]
            return real_ml + 1j * imag_ml
        else:
            return uniform_filter(cropped, size=window)[::win_y, ::win_x]

    def adaptive_filter(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        window_size: int = 7
    ) -> np.ndarray:
        half = window_size // 2
        padded_phase = np.pad(wrapped_phase, half, mode='reflect')
        padded_coh = np.pad(coherence, half, mode='reflect')

        filtered = np.zeros_like(wrapped_phase)
        h, w = wrapped_phase.shape

        y_grid, x_grid = np.mgrid[0:h, 0:w]

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                phase_shifted = padded_phase[half + dy:half + dy + h, half + dx:half + dx + w]
                coh_shifted = padded_coh[half + dy:half + dy + h, half + dx:half + dx + w]

                phase_diff = phase_shifted - wrapped_phase
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

                weight = coh_shifted / (np.abs(phase_diff) + 0.1)
                filtered += weight * phase_shifted

        weight_sum = np.zeros_like(wrapped_phase)
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                coh_shifted = padded_coh[half + dy:half + dy + h, half + dx:half + dx + w]
                phase_shifted = padded_phase[half + dy:half + dy + h, half + dx:half + dx + w]
                phase_diff = phase_shifted - wrapped_phase
                phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi
                weight = coh_shifted / (np.abs(phase_diff) + 0.1)
                weight_sum += weight

        filtered = filtered / (weight_sum + 1e-10)

        return filtered

    def snow_noise_removal(
        self,
        coherence: np.ndarray,
        min_coherence: float = 0.2,
        min_region_size: int = 50
    ) -> np.ndarray:
        from scipy.ndimage import label

        low_coh_mask = coherence < min_coherence

        labeled, num_features = label(low_coh_mask)

        cleaned = coherence.copy()

        for i in range(1, num_features + 1):
            region = labeled == i
            region_size = np.sum(region)

            if region_size < min_region_size:
                nearby_coh = np.median(coherence[~low_coh_mask])
                cleaned[region] = nearby_coh

        return cleaned

    def inject_atmospheric_phase(
        self,
        ifg_result,
        aps_phase: np.ndarray
    ):
        """
        向已有的干涉图结果注入合成大气相位屏幕(APS)
        物理上相当于电磁波穿过对流层时产生的相位延迟
        """
        H, W = ifg_result.wrapped_phase.shape
        assert aps_phase.shape == (H, W)

        interferogram_complex = ifg_result.interferogram
        phase_rotation = np.exp(1j * aps_phase)
        interferogram_with_aps = interferogram_complex * phase_rotation

        wrapped_phase_aps = np.angle(interferogram_with_aps)

        ifg_result.wrapped_phase = wrapped_phase_aps
        ifg_result.interferogram = interferogram_with_aps
        ifg_result.aps_phase = aps_phase

        return ifg_result
