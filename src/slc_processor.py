import numpy as np
from scipy import fft
from scipy.ndimage import map_coordinates, gaussian_filter
import time
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class CoregistrationResult:
    offset_x: float
    offset_y: float
    correlation_peak: float
    registered_image: np.ndarray
    cross_correlation: np.ndarray


class SLCProcessor:
    def __init__(self, width: int = 2048, height: int = 2048):
        self.width = width
        self.height = height
        self.master_image: Optional[np.ndarray] = None
        self.slave_image: Optional[np.ndarray] = None

    def generate_polar_slc(
        self,
        seed: int = 42,
        glacier_pattern: bool = True,
        noise_level: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(seed)

        y, x = np.mgrid[0:self.height, 0:self.width]

        amplitude = np.ones((self.height, self.width), dtype=np.float64)

        if glacier_pattern:
            num_glaciers = 8
            for _ in range(num_glaciers):
                cx = np.random.randint(0, self.width)
                cy = np.random.randint(0, self.height)
                radius = np.random.randint(100, 400)
                amp = np.random.uniform(1.5, 3.0)

                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                glacier = amp * np.exp(-dist ** 2 / (2 * radius ** 2))
                amplitude += glacier

            for _ in range(20):
                x0 = np.random.randint(0, self.width)
                y0 = np.random.randint(0, self.height)
                angle = np.random.uniform(0, np.pi)
                length = np.random.randint(200, 800)
                width_crevice = np.random.randint(5, 20)

                dx = x - x0
                dy = y - y0
                rot_x = dx * np.cos(angle) + dy * np.sin(angle)
                rot_y = -dx * np.sin(angle) + dy * np.cos(angle)

                crevice = np.where(
                    (np.abs(rot_x) < length / 2) & (np.abs(rot_y) < width_crevice),
                    -0.5,
                    0
                )
                amplitude += crevice

        speckle = np.random.exponential(1.0, (self.height, self.width))
        amplitude *= speckle

        phase = np.random.uniform(-np.pi, np.pi, (self.height, self.width))
        phase = gaussian_filter(phase, sigma=3)

        phase += 0.1 * np.sin(x / 50) * np.cos(y / 50)
        phase += 0.05 * np.sin(x / 200 + y / 150)

        complex_image = amplitude * np.exp(1j * phase)

        return complex_image, amplitude

    def generate_slave_from_master(
        self,
        master: np.ndarray,
        offset_x: float = 15.7,
        offset_y: float = -8.3,
        deformation_phase: float = 2.5,
        seed: int = 123
    ) -> np.ndarray:
        np.random.seed(seed)

        height, width = master.shape
        y, x = np.mgrid[0:height, 0:width]

        x_shifted = x - offset_x
        y_shifted = y - offset_y

        real_part = map_coordinates(
            np.real(master),
            [y_shifted.ravel(), x_shifted.ravel()],
            order=3,
            mode='reflect'
        ).reshape(height, width)

        imag_part = map_coordinates(
            np.imag(master),
            [y_shifted.ravel(), x_shifted.ravel()],
            order=3,
            mode='reflect'
        ).reshape(height, width)

        shifted = real_part + 1j * imag_part

        y_deform, x_deform = np.mgrid[0:height, 0:width]
        deform_phase = deformation_phase * np.sin(
            (x_deform - width / 2) / 300
        ) * np.cos(
            (y_deform - height / 2) / 400
        )
        deform_phase += 0.5 * np.random.normal(0, 0.3, (height, width))
        deform_phase = gaussian_filter(deform_phase, sigma=5)

        shifted = shifted * np.exp(1j * deform_phase)

        amp_noise = np.random.exponential(0.95, (height, width))
        phase_noise = np.random.normal(0, 0.15, (height, width))
        shifted = shifted * amp_noise * np.exp(1j * phase_noise)

        snow_mask = np.random.random((height, width)) < 0.05
        snow_mask = gaussian_filter(snow_mask.astype(float), sigma=10) > 0.3
        shifted[snow_mask] *= np.random.exponential(0.1, snow_mask.sum()) * np.exp(
            1j * np.random.uniform(-np.pi, np.pi, snow_mask.sum())
        )

        return shifted

    def frequency_domain_coregistration(
        self,
        master: np.ndarray,
        slave: np.ndarray,
        upsample_factor: int = 100
    ) -> CoregistrationResult:
        start_time = time.time()

        print("  [频域配准] 开始 FFT 变换...")
        master_fft = fft.fft2(master)
        slave_fft = fft.fft2(slave)

        print("  [频域配准] 计算互功率谱...")
        cross_power = master_fft * np.conj(slave_fft)
        cross_power_norm = cross_power / (np.abs(cross_power) + 1e-10)

        print("  [频域配准] 逆 FFT 得到互相关...")
        cross_correlation = np.abs(fft.ifft2(cross_power_norm))

        cross_correlation = fft.fftshift(cross_correlation)

        print("  [频域配准] 寻找粗匹配峰值...")
        peak_idx = np.unravel_index(
            np.argmax(cross_correlation),
            cross_correlation.shape
        )
        peak_y, peak_x = peak_idx

        center_y, center_x = self.height // 2, self.width // 2
        offset_y_coarse = peak_y - center_y
        offset_x_coarse = peak_x - center_x

        print(f"  [频域配准] 粗匹配偏移: dx={offset_x_coarse}, dy={offset_y_coarse}")
        print(f"  [频域配准] 峰值相关系数: {cross_correlation[peak_y, peak_x]:.4f}")

        print("  [频域配准] 亚像素精配 (Sinc 插值)...")
        peak_val, sub_px_offset = self._subpixel_refinement(
            cross_correlation,
            peak_y,
            peak_x,
            upsample_factor
        )

        offset_y = offset_y_coarse + sub_px_offset[0]
        offset_x = offset_x_coarse + sub_px_offset[1]

        print(f"  [频域配准] 亚像素偏移: dx={offset_x:.3f}, dy={offset_y:.3f}")

        print("  [频域配准] 重采样从影像...")
        registered = self._shift_image(slave, -offset_x, -offset_y)

        elapsed = time.time() - start_time
        print(f"  [频域配准] 完成! 耗时: {elapsed:.2f}s")

        return CoregistrationResult(
            offset_x=offset_x,
            offset_y=offset_y,
            correlation_peak=peak_val,
            registered_image=registered,
            cross_correlation=cross_correlation
        )

    def _subpixel_refinement(
        self,
        correlation: np.ndarray,
        peak_y: int,
        peak_x: int,
        upsample_factor: int = 100
    ) -> Tuple[float, Tuple[float, float]]:
        window_size = 4
        y_min = max(0, peak_y - window_size)
        y_max = min(correlation.shape[0], peak_y + window_size + 1)
        x_min = max(0, peak_x - window_size)
        x_max = min(correlation.shape[1], peak_x + window_size + 1)

        window = correlation[y_min:y_max, x_min:x_max]

        new_h = window.shape[0] * upsample_factor
        new_w = window.shape[1] * upsample_factor

        window_fft = fft.fft2(window)

        pad_h = new_h - window.shape[0]
        pad_w = new_w - window.shape[1]

        padded_fft = np.pad(
            window_fft,
            ((0, pad_h), (0, pad_w)),
            mode='constant'
        )

        upsampled = np.abs(fft.ifft2(padded_fft)) * (upsample_factor ** 2)

        upsampled_peak = np.unravel_index(np.argmax(upsampled), upsampled.shape)
        peak_val = upsampled[upsampled_peak]

        sub_y = upsampled_peak[0] / upsample_factor - window_size
        sub_x = upsampled_peak[1] / upsample_factor - window_size

        return peak_val, (sub_y, sub_x)

    def _shift_image(
        self,
        image: np.ndarray,
        shift_x: float,
        shift_y: float
    ) -> np.ndarray:
        height, width = image.shape
        y, x = np.mgrid[0:height, 0:width]

        x_shifted = x - shift_x
        y_shifted = y - shift_y

        coords = np.vstack([y_shifted.ravel(), x_shifted.ravel()])

        real_shifted = map_coordinates(
            np.real(image),
            coords,
            order=3,
            mode='reflect'
        ).reshape(height, width)

        imag_shifted = map_coordinates(
            np.imag(image),
            coords,
            order=3,
            mode='reflect'
        ).reshape(height, width)

        return real_shifted + 1j * imag_shifted

    def amplitude_image(self, complex_image: np.ndarray) -> np.ndarray:
        return np.abs(complex_image)

    def phase_image(self, complex_image: np.ndarray) -> np.ndarray:
        return np.angle(complex_image)

    def log_amplitude(self, complex_image: np.ndarray) -> np.ndarray:
        amp = np.abs(complex_image)
        return 20 * np.log10(amp + 1e-10)
