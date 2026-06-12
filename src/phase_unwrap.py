import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, label
from typing import Optional, Tuple, List
import time
from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass
class UnwrapResult:
    unwrapped_phase: np.ndarray
    quality_map: np.ndarray
    num_residues: int
    quality_mean: float
    algorithm: str = ""
    branch_cuts: Optional[np.ndarray] = None
    num_branch_cuts: int = 0
    unwrap_mask: Optional[np.ndarray] = None


class PhaseUnwrapper:
    def __init__(self, quality_threshold: float = 0.3):
        self.quality_threshold = quality_threshold

    def detect_residues(self, wrapped_phase: np.ndarray) -> np.ndarray:
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

    def branch_cut_unwrap(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        max_branch_length: int = 50
    ) -> UnwrapResult:
        start_time = time.time()
        print("  [枝切法解缠] 开始 Goldstein 枝切法相位解缠...")

        quality = coherence.copy()
        quality = gaussian_filter(quality, sigma=0.5)

        print("  [枝切法解缠] 步骤 1: 检测残差点...")
        residues = self.detect_residues(wrapped_phase)
        num_pos = np.sum(residues == 1)
        num_neg = np.sum(residues == -1)
        total_residues = num_pos + num_neg
        print(f"  [枝切法解缠]   正残差: {num_pos}, 负残差: {num_neg}, 总计: {total_residues}")

        if total_residues == 0:
            print("  [枝切法解缠]   无残差点，直接行列解缠...")
            unwrapped = self._row_column_unwrap(wrapped_phase, quality > self.quality_threshold)
            return UnwrapResult(
                unwrapped_phase=unwrapped,
                quality_map=quality,
                num_residues=0,
                quality_mean=float(np.mean(quality)),
                algorithm="Goldstein Branch-Cut",
                branch_cuts=np.zeros_like(residues, dtype=bool),
                num_branch_cuts=0,
                unwrap_mask=quality > self.quality_threshold
            )

        print("  [枝切法解缠] 步骤 2: 生成枝切线...")
        branch_cuts = self._generate_branch_cuts(
            residues, quality, max_branch_length
        )
        num_bc = np.sum(branch_cuts)
        print(f"  [枝切法解缠]   枝切线像素数: {num_bc}")

        print("  [枝切法解缠] 步骤 3: 洪水填充式解缠...")
        unwrapped, unwrap_mask = self._flood_fill_unwrap(
            wrapped_phase, quality, branch_cuts, residues
        )

        mean_quality = np.mean(quality)
        elapsed = time.time() - start_time

        print(f"  [枝切法解缠] 完成! 耗时: {elapsed:.2f}s")
        print(f"  [枝切法解缠]   解缠像素: {np.sum(unwrap_mask)}/{unwrap_mask.size} ({np.sum(unwrap_mask)/unwrap_mask.size:.1%})")

        return UnwrapResult(
            unwrapped_phase=unwrapped,
            quality_map=quality,
            num_residues=int(total_residues),
            quality_mean=float(mean_quality),
            algorithm="Goldstein Branch-Cut",
            branch_cuts=branch_cuts,
            num_branch_cuts=int(num_bc),
            unwrap_mask=unwrap_mask
        )

    def _generate_branch_cuts(
        self,
        residues: np.ndarray,
        quality: np.ndarray,
        max_length: int = 50
    ) -> np.ndarray:
        h, w = residues.shape
        branch_cuts = np.zeros((h, w), dtype=bool)

        pos_residues = list(zip(*np.where(residues == 1)))
        neg_residues = list(zip(*np.where(residues == -1)))

        all_residues = pos_residues + neg_residues
        visited = set()

        for start_r, start_c in all_residues:
            if (start_r, start_c) in visited:
                continue

            charge = residues[start_r, start_c]
            if charge == 0:
                continue

            visited.add((start_r, start_c))

            priority_queue = []
            heapq.heappush(priority_queue, (0, start_r, start_c, charge, [(start_r, start_c)]))

            found = False
            best_path = []
            best_cost = float('inf')

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]

            visited_in_search = set()
            visited_in_search.add((start_r, start_c))

            while priority_queue and not found:
                cost, r, c, current_charge, path = heapq.heappop(priority_queue)

                if len(path) > max_length:
                    continue

                if current_charge == 0 and len(path) > 1:
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
                        found = True
                    continue

                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited_in_search:
                        new_charge = current_charge + residues[nr, nc]

                        dist = np.sqrt(dr * dr + dc * dc)
                        qual = quality[min(nr, quality.shape[0]-1), min(nc, quality.shape[1]-1)]
                        step_cost = dist * (2.0 - qual)

                        new_cost = cost + step_cost
                        new_path = path + [(nr, nc)]

                        heapq.heappush(priority_queue, (new_cost, nr, nc, new_charge, new_path))
                        visited_in_search.add((nr, nc))

            if best_path:
                for r, c in best_path:
                    branch_cuts[r, c] = True
                    if residues[r, c] != 0:
                        visited.add((r, c))
            else:
                for r, c in path:
                    branch_cuts[r, c] = True

        return branch_cuts

    def _flood_fill_unwrap(
        self,
        wrapped_phase: np.ndarray,
        quality: np.ndarray,
        branch_cuts: np.ndarray,
        residues: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = wrapped_phase.shape
        unwrapped = np.zeros_like(wrapped_phase)
        unwrap_mask = np.zeros((h, w), dtype=bool)

        valid_mask = quality > self.quality_threshold

        best_quality = -1
        start_r, start_c = 0, 0
        for i in range(h):
            for j in range(w):
                if valid_mask[i, j] and quality[i, j] > best_quality:
                    on_branch = False
                    if i < branch_cuts.shape[0] and j < branch_cuts.shape[1]:
                        if branch_cuts[i, j]:
                            on_branch = True
                    if not on_branch:
                        best_quality = quality[i, j]
                        start_r, start_c = i, j

        if best_quality < 0:
            return wrapped_phase.copy(), np.zeros((h, w), dtype=bool)

        unwrapped[start_r, start_c] = wrapped_phase[start_r, start_c]
        unwrap_mask[start_r, start_c] = True

        border_pixels = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < h and 0 <= nc < w and valid_mask[nr, nc]:
                cross_branch = self._crosses_branch_cut(start_r, start_c, nr, nc, branch_cuts)
                if not cross_branch:
                    qual = quality[nr, nc]
                    heapq.heappush(border_pixels, (-qual, nr, nc, start_r, start_c))

        while border_pixels:
            neg_qual, r, c, ref_r, ref_c = heapq.heappop(border_pixels)

            if unwrap_mask[r, c]:
                continue

            phase_diff = wrapped_phase[r, c] - wrapped_phase[ref_r, ref_c]
            phase_diff = np.mod(phase_diff + np.pi, 2 * np.pi) - np.pi

            unwrapped[r, c] = unwrapped[ref_r, ref_c] + phase_diff
            unwrap_mask[r, c] = True

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if (0 <= nr < h and 0 <= nc < w and
                    not unwrap_mask[nr, nc] and valid_mask[nr, nc]):
                    cross_branch = self._crosses_branch_cut(r, c, nr, nc, branch_cuts)
                    if not cross_branch:
                        qual = quality[nr, nc]
                        heapq.heappush(border_pixels, (-qual, nr, nc, r, c))

        for i in range(h):
            for j in range(w):
                if not unwrap_mask[i, j]:
                    unwrapped[i, j] = wrapped_phase[i, j]

        return unwrapped, unwrap_mask

    def _crosses_branch_cut(
        self,
        r1: int, c1: int,
        r2: int, c2: int,
        branch_cuts: np.ndarray
    ) -> bool:
        h, w = branch_cuts.shape
        mid_r = (r1 + r2) // 2
        mid_c = (c1 + c2) // 2
        if 0 <= mid_r < h and 0 <= mid_c < w:
            if branch_cuts[mid_r, mid_c]:
                return True
        return False

    def network_flow_unwrap(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray,
        use_quality: bool = True
    ) -> UnwrapResult:
        start_time = time.time()
        print("  [网络流解缠] 开始最小费用网络流相位解缠...")

        quality = coherence.copy()
        quality = gaussian_filter(quality, sigma=0.5)

        print("  [网络流解缠] 步骤 1: 检测残差点...")
        residues = self.detect_residues(wrapped_phase)
        num_pos = np.sum(residues == 1)
        num_neg = np.sum(residues == -1)
        total_residues = num_pos + num_neg
        print(f"  [网络流解缠]   正残差: {num_pos}, 负残差: {num_neg}, 总计: {total_residues}")

        if total_residues == 0:
            print("  [网络流解缠]   无残差点，直接行列解缠...")
            unwrapped = self._row_column_unwrap(wrapped_phase, quality > self.quality_threshold)
            return UnwrapResult(
                unwrapped_phase=unwrapped,
                quality_map=quality,
                num_residues=0,
                quality_mean=float(np.mean(quality)),
                algorithm="Minimum Cost Network Flow",
                branch_cuts=np.zeros_like(residues, dtype=bool),
                num_branch_cuts=0,
                unwrap_mask=quality > self.quality_threshold
            )

        print("  [网络流解缠] 步骤 2: 构建网络流图...")
        h, w = residues.shape
        valid_mask = quality[:h, :w] > self.quality_threshold * 0.5

        pos_nodes = list(zip(*np.where((residues == 1) & valid_mask)))
        neg_nodes = list(zip(*np.where((residues == -1) & valid_mask)))

        print(f"  [网络流解缠]   有效正残差: {len(pos_nodes)}, 有效负残差: {len(neg_nodes)}")

        if len(pos_nodes) == 0 or len(neg_nodes) == 0:
            print("  [网络流解缠]   单侧残差不足，回退到行列解缠...")
            unwrapped = self._row_column_unwrap(wrapped_phase, quality > self.quality_threshold)
            return UnwrapResult(
                unwrapped_phase=unwrapped,
                quality_map=quality,
                num_residues=int(total_residues),
                quality_mean=float(np.mean(quality)),
                algorithm="Minimum Cost Network Flow (fallback)",
                branch_cuts=np.zeros_like(residues, dtype=bool),
                num_branch_cuts=0,
                unwrap_mask=quality > self.quality_threshold
            )

        print("  [网络流解缠] 步骤 3: 最小费用匹配 (匈牙利近似)...")
        cuts, flow_cost = self._min_cost_matching(
            pos_nodes, neg_nodes, quality, use_quality
        )

        branch_cuts_img = np.zeros_like(residues, dtype=bool)
        for (r1, c1), (r2, c2) in cuts:
            self._draw_line(branch_cuts_img, r1, c1, r2, c2)

        num_bc = np.sum(branch_cuts_img)
        print(f"  [网络流解缠]   匹配对数: {len(cuts)}, 总费用: {flow_cost:.2f}")
        print(f"  [网络流解缠]   枝切线像素数: {num_bc}")

        print("  [网络流解缠] 步骤 4: 质量引导解缠...")
        unwrapped, unwrap_mask = self._flood_fill_unwrap(
            wrapped_phase, quality, branch_cuts_img, residues
        )

        mean_quality = np.mean(quality)
        elapsed = time.time() - start_time

        print(f"  [网络流解缠] 完成! 耗时: {elapsed:.2f}s")
        print(f"  [网络流解缠]   解缠像素: {np.sum(unwrap_mask)}/{unwrap_mask.size} ({np.sum(unwrap_mask)/unwrap_mask.size:.1%})")

        return UnwrapResult(
            unwrapped_phase=unwrapped,
            quality_map=quality,
            num_residues=int(total_residues),
            quality_mean=float(mean_quality),
            algorithm="Minimum Cost Network Flow",
            branch_cuts=branch_cuts_img,
            num_branch_cuts=int(num_bc),
            unwrap_mask=unwrap_mask
        )

    def _min_cost_matching(
        self,
        pos_nodes: List[Tuple[int, int]],
        neg_nodes: List[Tuple[int, int]],
        quality: np.ndarray,
        use_quality: bool = True,
        max_residues: int = 500
    ) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], float]:
        pos_arr = np.array(pos_nodes)
        neg_arr = np.array(neg_nodes)

        if len(pos_nodes) > max_residues or len(neg_nodes) > max_residues:
            print(f"  [网络流解缠]   残差过多 ({len(pos_nodes)}+{len(neg_nodes)})，稀疏化到 {max_residues} 对...")
            pos_sampled = self._spatial_sparsify(pos_arr, max_residues)
            neg_sampled = self._spatial_sparsify(neg_arr, max_residues)
            pos_nodes = [tuple(p) for p in pos_sampled]
            neg_nodes = [tuple(p) for p in neg_sampled]

        n_pos = len(pos_nodes)
        n_neg = len(neg_nodes)

        matches = []
        total_cost = 0.0
        used_neg = set()

        pos_arr = np.array(pos_nodes)
        neg_arr = np.array(neg_nodes)

        for i in range(n_pos):
            r1, c1 = pos_nodes[i]

            if len(used_neg) == n_neg:
                break

            dists = np.sqrt((neg_arr[:, 0] - r1) ** 2 + (neg_arr[:, 1] - c1) ** 2)

            if use_quality:
                mid_r = (neg_arr[:, 0] + r1) // 2
                mid_c = (neg_arr[:, 1] + c1) // 2
                mid_r = np.clip(mid_r, 0, quality.shape[0] - 1)
                mid_c = np.clip(mid_c, 0, quality.shape[1] - 1)
                avg_quality = quality[mid_r, mid_c]
                costs = dists * (2.0 - avg_quality)
            else:
                costs = dists

            for j in used_neg:
                costs[j] = np.inf

            best_j = int(np.argmin(costs))
            min_cost = costs[best_j]

            if np.isfinite(min_cost):
                matches.append((pos_nodes[i], neg_nodes[best_j]))
                total_cost += min_cost
                used_neg.add(best_j)

        return matches, total_cost

    def _spatial_sparsify(
        self,
        points: np.ndarray,
        target_count: int
    ) -> np.ndarray:
        if len(points) <= target_count:
            return points

        h_min, h_max = np.min(points[:, 0]), np.max(points[:, 0])
        w_min, w_max = np.min(points[:, 1]), np.max(points[:, 1])

        grid_size = int(np.sqrt(target_count))
        cell_h = (h_max - h_min + 1) / grid_size
        cell_w = (w_max - w_min + 1) / grid_size

        sampled = []
        for i in range(grid_size):
            for j in range(grid_size):
                h_low = h_min + i * cell_h
                h_high = h_min + (i + 1) * cell_h
                w_low = w_min + j * cell_w
                w_high = w_min + (j + 1) * cell_w

                in_cell = (
                    (points[:, 0] >= h_low) & (points[:, 0] < h_high) &
                    (points[:, 1] >= w_low) & (points[:, 1] < w_high)
                )
                if np.any(in_cell):
                    idx = np.where(in_cell)[0]
                    sampled.append(points[idx[0]])

        if len(sampled) == 0:
            step = len(points) // target_count
            sampled = points[::step][:target_count]

        return np.array(sampled)

    def _draw_line(
        self,
        img: np.ndarray,
        r1: int, c1: int,
        r2: int, c2: int
    ):
        h, w = img.shape
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc

        r, c = r1, c1
        while True:
            if 0 <= r < h and 0 <= c < w:
                img[r, c] = True

            if r == r2 and c == c2:
                break

            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _row_column_unwrap(
        self,
        wrapped_phase: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        h, w = wrapped_phase.shape
        unwrapped = np.copy(wrapped_phase)

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

    def quality_guided_unwrap(
        self,
        wrapped_phase: np.ndarray,
        coherence: np.ndarray
    ) -> UnwrapResult:
        start_time = time.time()
        print("  [质量引导解缠] 开始质量引导相位解缠...")

        quality = coherence.copy()
        quality = gaussian_filter(quality, sigma=1)

        print("  [质量引导解缠] 检测残差点...")
        residues = self.detect_residues(wrapped_phase)
        num_residues = np.sum(np.abs(residues))
        print(f"  [质量引导解缠] 残差点数量: {int(num_residues)}")

        print("  [质量引导解缠] 执行行列解缠...")
        mask = quality > self.quality_threshold
        unwrapped = self._row_column_unwrap(wrapped_phase, mask)

        mean_quality = np.mean(quality)
        elapsed = time.time() - start_time
        print(f"  [质量引导解缠] 完成! 耗时: {elapsed:.2f}s")

        return UnwrapResult(
            unwrapped_phase=unwrapped,
            quality_map=quality,
            num_residues=int(num_residues),
            quality_mean=float(mean_quality),
            algorithm="Quality Guided",
            unwrap_mask=mask
        )

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
