import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from typing import Optional
import time
from dataclasses import dataclass


@dataclass
class UnwrapResult:
    unwrapped_phase: np.ndarray
    quality_map: np.ndarray
    num_residues: int
    quality_mean: float


class PhaseUnwrapper:
    def __init__(self, quality_threshold: float = 0.3):
        self.quality_threshold = quality_threshold

    def quality_guided_unwrap(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray
    ) -> UnwrapResult:
        start_time = time.time()
        print("  [相位解缠] 质量引导相位解缠...")

        quality = coherence.copy()
        quality = gaussian_filter(quality, sigma=1)

        print("  [相位解缠] 检测残差点...")
        residues = self._detect_residues(wrapped_phase)
        num_residues = np.sum(np.abs(residues))
        print(f"  [相位解缠] 残差点数量: {int(num_residues)}")

        print("  [相位解缠] 执行质量引导解缠...")
        unwrapped = self._quality_algorithm(wrapped_phase, quality)

        mean_quality = np.mean(quality)

        elapsed = time.time() - start_time
        print(f"  [相位解缠] 完成! 耗时: {elapsed:.2f}s")

        return UnwrapResult(
            unwrapped_phase=unwrapped,
            quality_map=quality,
            num_residues=int(num_residues),
            quality_mean=float(mean_quality)
        )

    def _detect_residues(self, wrapped_phase: np.ndarray) -> np.ndarray:
        h, w = wrapped_phase.shape
        residues = np.zeros((h - 1, w - 1), dtype=np.int8)

        d1 = wrapped_phase[:-1, :-1]
        d2 = wrapped_phase[:-1, 1:]
        d3 = wrapped_phase[1:, 1:]
        d4 = wrapped_phase[1:, :-1]

        phase_diff1 = np.mod(d2 - d1 + np.pi, 2 * np.pi) - np.pi
        phase_diff2 = np.mod(d3 - d2 + np.pi, 2 * np.pi) - np.pi
        phase_diff3 = np.mod(d4 - d3 + np.pi, 2 * np.pi) - np.pi
        phase_diff4 = np.mod(d1 - d4 + np.pi, 2 * np.pi) - np.pi

        sum_diff = phase_diff1 + phase_diff2 + phase_diff3 + phase_diff4
        residues = np.round(sum_diff / (2 * np.pi)).astype(np.int8)

        return residues

    def _quality_algorithm(
        self,
        wrapped_phase: np.ndarray,
        quality: np.ndarray
    ) -> np.ndarray:
        h, w = wrapped_phase.shape
        unwrapped = np.copy(wrapped_phase)
        mask = quality > self.quality_threshold

        if not np.any(mask):
            return wrapped_phase

        for i in range(h):
            for j in range(1, w):
                if mask[i, j] and mask[i, j-1]:
                    diff = wrapped_phase[i, j] - wrapped_phase[i, j-1]
                    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                    unwrapped[i, j] = unwrapped[i, j-1] + diff

        for j in range(w):
            for i in range(1, h):
                if mask[i, j] and mask[i-1, j]:
                    diff = wrapped_phase[i, j] - wrapped_phase[i-1, j]
                    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
                    unwrapped[i, j] = unwrapped[i-1, j] + diff

        return unwrapped

    def simple_unwrap(self, wrapped_phase: np.ndarray) -> np.ndarray:
        unwrapped = np.copy(wrapped_phase)

        for i in range(unwrapped.shape[0]):
            for j in range(1, unwrapped.shape[1]):
                diff = unwrapped[i, j] - unwrapped[i, j-1]
                if diff > np.pi:
                    unwrapped[i, j:] -= 2 * np.pi
                elif diff < -np.pi:
                    unwrapped[i, j:] += 2 * np.pi

        for j in range(unwrapped.shape[1]):
            for i in range(1, unwrapped.shape[0]):
                diff = unwrapped[i, j] - unwrapped[i-1, j]
                if diff > np.pi:
                    unwrapped[i:, j] -= 2 * np.pi
                elif diff < -np.pi:
                    unwrapped[i:, j] += 2 * np.pi

        return unwrapped

    def phase_to_displacement(
        self,
        unwrapped_phase: np.ndarray,
        wavelength: float = 0.056,
        incidence_angle: float = 39.0
    ) -> np.ndarray:
        inc_rad = np.radians(incidence_angle)
        displacement = (unwrapped_phase * wavelength) / (4 * np.pi * np.cos(inc_rad))
        return displacement

    def spatial_filter(
        self,
        phase: np.ndarray,
        coherence: Optional[np.ndarray] = None,
        method: str = 'gaussian',
        sigma: float = 2.0
    ) -> np.ndarray:
        if method == 'gaussian':
            return gaussian_filter(phase, sigma=sigma)
        elif method == 'median':
            return median_filter(phase, size=int(sigma * 2 + 1))
        elif method == 'adaptive':
            if coherence is None:
                return gaussian_filter(phase, sigma=sigma)

            s = int(sigma * 3)
            padded = np.pad(phase, s, mode='reflect')
            coh_padded = np.pad(coherence, s, mode='reflect')

            h, w = phase.shape
            filtered = np.zeros_like(phase)
            total_weight = np.zeros_like(phase)

            for dy in range(-s, s + 1):
                for dx in range(-s, s + 1):
                    dist = np.sqrt(dy ** 2 + dx ** 2)
                    if dist > sigma * 3:
                        continue

                    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
                    coh_weight = coh_padded[s + dy:s + dy + h, s + dx:s + dx + w]
                    weight *= coh_weight

                    shifted = padded[s + dy:s + dy + h, s + dx:s + dx + w]
                    phase_diff = shifted - phase
                    phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

                    filtered += weight * (phase + phase_diff)
                    total_weight += weight

            filtered = filtered / (total_weight + 1e-10)
            return filtered
        else:
            return phase
